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

### What each row reads

Every row is read for the keys below and no others. A key nothing reads is
refused by name, because otherwise it would load and be discarded: `alias:`
written for `aliases:` would add no alias and report nothing. Each value is
checked for its shape, not converted into one. A key written as `null` reads
as absent. Any refusal is a `ValidationError` naming the section, the row and
the key.

The shapes the tables share:

- **A label** (`name`, `description`, `unit`) is a string. `name: 2024` is
  refused rather than rendered: quote it.
- **An alias list** is a list of non-empty strings. `aliases: Beagle` is
  refused, because a bare string would be read one character at a time. One
  alias is a list of one: `aliases: [Beagle]`.
- **A flag** is `true` or `false`. `"no"` is refused rather than read as
  truthy.
- **A mapping** is a mapping. A list of pairs is refused.
- **A reference** (`isa`, `inverse_of`) names one id. A number names the id it
  renders as, since ids are read the same way, so `isa: 100` names
  `id: 100`.

| Section | Required | Optional |
|---|---|---|
| `entity_types:` | `id` | `name`, `description`, `aliases`, `metadata`, `attributes` (below), `isa` |
| `relation_types:` | `id` | `name`, `description`, `aliases`, `metadata`, `domain`, `range`, `inverse_of`, `symmetric`, `transitive`, `inference` (`on_demand` or `materialized`) |
| `entities:` | `id`, `type` | `name`, `description`, `aliases`, `metadata`, `source` |
| `assertions:` | `subject`, `relation`, `object` | `id` (minted from the edge when absent), `metadata`, `polarity` (`asserted` or `negated`) |
| `taxonomies:` | `id`, `relation` | `name`, `description`, `metadata`, `materialization` (`structure` and `content`, each `on_demand` or `materialized`) |

Three of those values take a further shape:

- **An `object:`** is a string naming an entity, a mapping `{entity: …}`, any
  other scalar as a literal, or a literal mapping read for `value`, `type`,
  `unit` and `metadata`. A literal's `type:` is a record field type, as an
  attribute's `field_type:` is, and is `string` when absent.
  `{value: 3.5, type: number}` is refused, since `number` names no record type.
- **A `source:`** reads `source_id`, `kind` (`record` when absent) and
  `projection_id`, each a string. **Its other keys are kept**, as the
  locator into your own data, so this is the one block with an open key set.
- **A `taxonomies:` row declaring `kind:`** asks for a backing another door
  binds, and that door reads its keys (see
  [Taxonomies and how a tree is projected](#taxonomies-and-how-a-tree-is-projected)).

`cardinality:`, `constraints:` and `condition:` are refused with a message of
their own, because a later version reads them.

A **nested tree's** nodes are your data, so they have no closed key set. Their
`aliases`, `description` and `metadata` are still read for their shape.

### What an attribute row reads

An attribute row is read for seven keys. Any other key is refused by name,
because an unread key would load and be discarded: `enum:` written for
`enum_values:` would enumerate nothing and report nothing. A key written as
`null` reads as absent.

| Key | Takes | Absent |
|---|---|---|
| `name` | a non-empty string, unique within its entity type | refused |
| `type` | a string: the vocabulary's type name, kept as written. The vocabulary is open, so `entity`, `enum` and `number` are all legal | `string` |
| `field_type` | a record field type — `string`, `text`, `integer`, `float`, `boolean`, `datetime`, `json`, `binary`, `vector`, `sparse_vector`, in any case, or the `FieldType` member itself. It **supplies** the record type for a `type:` that names none, and must agree with one that does | derived from `type:`, so an attribute with no `type:` is `string` |
| `entity_type` | a non-empty string: the id of a declared entity type, for an `entity`-typed attribute | none |
| `required` | `true` or `false`. `"no"` is refused rather than read as truthy | `false` |
| `enum_values` | a non-empty list of strings, each named once. An empty list would allow no value at all | not enumerated |
| `description` | a string, which is what an extraction prompt is built from | `""` |

`field_type:` is for a vocabulary type that has no record counterpart of its
own. `number` is one. Beside a `type:` that is already a record type it can
only restate it: `{type: integer, field_type: string}` is refused, because no
reading honours both.

```python
from dataknobs_common.fields import FieldType
from dataknobs_common.ontology import load_ontology

vocabulary = load_ontology(
    {
        "id": "zoo",
        "entity_types": [
            {
                "id": "Species",
                "attributes": [{"name": "lifespan_years", "type": "number", "field_type": "float"}],
            }
        ],
    }
)
(lifespan,) = vocabulary.inherited_attributes("Species")
assert (lifespan.value_type, lifespan.field_type) == ("number", FieldType.FLOAT)
```

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

The fields you read most are sources rather than containers:

| Field | What it is |
|---|---|
| `id`, `version` | the vocabulary's own identity |
| `entity_types`, `relation_types` | the declared kinds, as `EntityType` and `RelationType` |
| `entities` | an `EntitySource` — `get`, `get_many`, `by_surface_form`, `by_type`, `fetch_origin`, `fetch_origins`, `describe` |
| `assertions` | an `AssertionSource` — `get`, `find`, `find_many` |
| `taxonomies` | `TaxonomyDefinition` per declared axis, keyed by its own id |
| `structures` | a materialized structure per axis, where one was asked for |
| `describes`, `imports` | what this vocabulary claims to describe, and what it pulls in |
| `variants` | what a `variants:` section admitted and refused, or `None` with no section — see [Variants of a form](#variants-of-a-form) |

The value types are frozen dataclasses: `Entity`, `EntityRef`, `Assertion`,
`AttributeDef`, `Term` (an `EntityRef` or a `Literal`). An `Assertion` carries a
`Polarity` — `ASSERTED` or `NEGATED` — so a vocabulary can state that something
is *not* the case, and a walk can honour it.

`Ontology` and `AsyncOntology` are frozen and compared by identity: they hold
the sources a vocabulary was built over, and two loads of one document are two
vocabularies rather than one repeated.

### A loaded vocabulary does not change

A source builds its lookups once, when it is constructed: every entity's id,
name and aliases folded into one map, every type into another, every assertion
indexed by subject and object. So what it hands out cannot change afterwards,
or `get()` would show one thing while the lookups answered from another.

- **The value types are frozen**: `Entity`, `EntityType`, `RelationType`,
  `AttributeDef`, `Assertion`, `Literal`, `Provenance`, `SourceRef` and
  `ResolutionRef`. Assigning a field raises `FrozenInstanceError`. They still
  compare field-wise, and are still unhashable.
- **Their collections are immutable.** `aliases`, `attributes` and
  `enum_values` are tuples; `metadata`, `SourceRef.locator` and
  `ResolutionRef.corpus` are a `FrozenMapping`, frozen all the way down. A list
  or dict handed to a constructor is converted, so a YAML row or a test passing
  one still works.
- **A source copies the container it is handed.** An entity added to the dict
  you built a `MappingEntitySource` from afterwards reaches none of its
  members, `get()` included.

Derive a changed entity with `dataclasses.replace`, which builds a new one:

```python
from dataclasses import replace

from dataknobs_common.ontology import Entity

beagle = Entity(id="beagle", type="Breed", aliases=["English Beagle"])
assert beagle.aliases == ("English Beagle",)

renamed = replace(beagle, aliases=(*beagle.aliases, "Snoopy"))
assert beagle.aliases == ("English Beagle",)   # the original is unchanged
```

**A list under `metadata:` reads back as a tuple.** The freeze reaches every
level, so compare against a tuple:

```python
entity = Entity(id="beagle", type="Breed", metadata={"tags": ["hound"]})
assert entity.metadata["tags"] == ("hound",)
assert entity.metadata["tags"] != ["hound"]    # a tuple never equals a list
```

Two entities built from the same document still compare equal, and two loads
of one document still answer the same
[revision](#which-state-of-a-vocabulary-answered), because both sides are frozen
the same way.

**Frozen is not JSON.** `json.dumps` takes a `dict` and a `list` and no other
mapping or sequence, so `json.dumps(entity.metadata)` raises `TypeError`, as
does dumping `dataclasses.asdict(entity)`, which keeps the mapping as it
found it. `thaw_structure` gives back plain dicts, lists and sets, nested
values included — `dict(entity.metadata)` thaws the top level only:

```python
import json

from dataknobs_common import thaw_structure

json.dumps(thaw_structure(entity.metadata))    # '{"tags": ["hound"]}'
```

**What frozen does not reach.** The containers are frozen; a leaf is held as
given. A `bytearray` or an object of your own inside `metadata`, or a
`Literal`'s `value`, can still change in place, and a revision digests what
such a leaf held when the source was built. Keep leaves immutable — the
ordinary case for anything read from YAML or JSON — or build a new entity
when one changes.

## Variants of a form

A reader types *apples* and the vocabulary says *Apple*; a catalogue writes
*Fruit & Veg* and the reader types *fruit and veg*. A fold cannot bridge either
-- it runs on every query, and must not change what a word is -- so the
variants are made once, where the vocabulary is bound, and each is written as
an alias of its entity. Opt in with a `variants:` section naming the rules, in
the order they compose:

```yaml
id: grocery
entity_types:
  - id: item
entities:
  - {id: apple, type: item, name: Apple}
  - {id: glass, type: item, name: Glass}
  - {id: glasses, type: item, name: Glasses}
  - {id: acme-pb, type: item, name: Acme Peanut Butter, metadata: {brand: Acme}}
variants:
  - drop_parentheticals
  - ampersand
  - slash_or
  - inflect
  - {rule: strip_prefix, field: brand}
```

```python
from dataknobs_common.ontology import load_ontology

onto = load_ontology(grocery)  # the document above, as a mapping or a path
onto.by_surface_form("apples")  # frozenset({"apple"})
onto.by_surface_form("peanut butter")  # frozenset({"acme-pb"})
onto.by_surface_form("glasses")  # frozenset({"glasses"}) -- and only that
[refusal.variant.form for refusal in onto.variants.refused]  # ["Glasses", "Glass"]
```

**Each rule sees what the rules before it made**, so *Fruit & Veg (fresh)*
reaches *Fruit and Veg* through `drop_parentheticals` and then `ampersand`.
The first four are `dataknobs_common.text.DEFAULT_RULES`, and they are
English: `inflect` makes the last word plural or singular by the regular
endings. A rule given parameters is a mapping naming it under `rule`;
`strip_prefix` reads its prefix from the entity's `metadata`.

**Regular endings misread a few words.** Admission refuses a collision between
entities, not a misreading: *News* would gain *New*, and *Jeans* *Jean*, which
are real words a reader types meaning something else. Name such words in
`inflect`'s `keep` -- `{rule: inflect, keep: [news, jeans]}` -- and they are
left as written. Most misreadings give a spelling nobody types (*Series* gives
*Sery*) and cost nothing.

| Rule | Example | In `DEFAULT_RULES` |
|---|---|---|
| `drop_parentheticals` | *Fruit & Veg (fresh)* → *Fruit & Veg* | yes |
| `ampersand` | *Fruit & Veg* → *Fruit and Veg* | yes |
| `slash_or` | *Jam / Jelly* → *Jam or Jelly*; not in a form holding a digit | yes |
| `inflect` (`keep`) | *Apple* → *Apples*; *Cherries* → *Cherry* | yes |
| `camelcase` | *BlueBerry* → *Blue Berry* | no |
| `hyphen` (`subs`) | *Gluten-Free* → *Gluten Free*, *GlutenFree* | no |
| `split` (`sep`, `min_token_len`) | *Jam / Jelly* → *Jam*, *Jelly* | no: makes fragments |
| `parenthetical` | *Soda (Pop)* → *Pop* | no: makes a fragment |
| `strip_prefix` (`field`) | *Acme Peanut Butter* → *Peanut Butter* | no: reads the entity's metadata |

**A variant that would make a collision is refused, not dropped.** *Glass*'s
plural is another entity's name, so it is not admitted, and neither is
*Glasses*'s singular. A variant may repeat an ambiguity its source already has
-- two entities both named *Apple* each get *Apples* -- but it may not create
one. Collisions are counted under the door's own fold, over every id, name and
alias the document declares. `onto.variants` holds each `Refusal` with the
entities it would have met; each load also logs the counts at INFO and every
refusal at DEBUG, on `dataknobs_common.ontology.variants`. An unknown rule
name is refused when the document is built, by name.

`OntologyRegistry` in `dataknobs-data` reads the section the same way. The
vocabulary's entities as authored are what a reload diffs, so changing the
rules is not reported as a change to the vocabulary.

**Rules of your own** are registered by name, and every configuration can then
name them. A registry entry is a factory taking the rule's parameters:

```python
from dataknobs_common.text import VARIANT_RULES

def pack_of(config):
    size = config["size"]
    return lambda form, context: (f"{form} {size}-pack",)

VARIANT_RULES.register("pack_of", pack_of)
# variants: [{rule: pack_of, size: 6}]
```

**In code**, `with_variants` does what a door does, for entities you hold, and
`dataknobs_common.text.admit` is the primitive under both, generic in whatever
your entities are keyed by:

```python
from dataknobs_common.ontology import Entity, MappingEntitySource, with_variants
from dataknobs_common.text import Declared, admit

entities = {"apple": Entity(id="apple", type="item", name="Apple")}
varied, report = with_variants(entities)  # the default rules
MappingEntitySource(varied).by_surface_form("apples")  # frozenset({"apple"})

report = admit([Declared(1, "Glass"), Declared(2, "Glasses")])
report.refused[0].others  # (2,)
```

**A live table is not varied for you.** A source over a table of your own
(`kind: record`, with a surface-form side table) is filled by you, and
`OntologyRegistry` refuses a `variants:` section beside one. Fill the side
table from `admit`'s `admitted` forms instead, and re-declare
`surface_forms.longest_form_tokens:` as the larger of your own widest form and
`report.longest_form_tokens()` -- `ampersand` and `slash_or` each add a token,
and a scan never looks wider than the declared bound.

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
rather than a key in the open `metadata` mapping.

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
projection, the `capabilities` the source has, the entity types it
`declares`, and its `form_edge`: how many punctuation characters the widest
edge of any declared form carries — `1` for `(beagle)` or `C.D.C.`, `0` when no
form has punctuation at either end, `None` when the source cannot say. A
scanning rung reads it to reach a form like `(beagle)` inside a sentence;
`MappingEntitySource` measures it over its forms, as written and as folded, and
a live table declares it. `AUTHORED_SOURCE_ID` (`"authored"`) is the source id a hand-written
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
member a conforming source may decline. (`revision` may answer `None`, which
is not declining: it is the answer, and it says the source cannot tell — see
[Which state of a vocabulary answered](#which-state-of-a-vocabulary-answered).) The fold is the **source's** —
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

### Which state of a vocabulary answered

A link you record against a vocabulary — a mention resolved to an entity, an
index built over its forms — is only as good as the vocabulary that made it.
`revision()` says which state of the vocabulary is answering, so you can store
it beside the link and ask again later: two different answers mean the
vocabulary has changed since, and the link is worth checking rather than
trusting.

```python
from dataknobs_common.ontology import load_ontology

document = {
    "id": "office",
    "version": "2.3",
    "entity_types": [{"id": "product"}],
    "entities": [{"id": "p-1", "type": "product", "name": "PowerPoint"}],
}
recorded = load_ontology(document).entities.revision()   # '2.3@3c42cd2a…'

# Later: the same document, now admitting camelCase variants.
current = load_ontology({**document, "variants": ["camelcase"]}).entities.revision()
assert current != recorded   # 'Power Point' is an alias now, so the link is worth a look
```

It is on both source protocols, awaitable on `AsyncEntitySource` because a
live source reads its answer from its owner. A source that can answer declares
`Capability.SOURCE_REVISION`; one that cannot answers `None`, and only such a
source does. An authored source always answers.

**Compare answers for equality, and do not parse them.** An authored source
spells its answer `<version>@<digest>`, where the version is the document's
(`1.0` when it declares none — that is `OntologyConfig`'s default, and what
`onto.version` reports) and the digest is a SHA-256 over the entities the
source holds, variants included. A source built in code with no `version=`
answers the bare digest. The `@` is not a separator you may split on: a
version string may contain one.

The digest is taken **when the source is built**, alongside the lookups, so
it describes the state the lookups answer from. A value in an entity that no
digest can identify — a lambda in its `metadata` — fails the construction with
a `TypeError` naming the entity and field. NaN does not: two documents that
both say `.nan` hold the same content, and answer the same revision.

Four things the answer does not track exactly, each erring in a stated
direction:

- **The fold.** A normalizer is a callable, and is not digested. The same
  entities built with a different `normalizer=` answer the same revision while
  resolving differently. A fold is code, and a fold change is a code change.
- **Values Python calls equal.** The digest encodes content as Python
  compares it, so `1`, `1.0` and `true` in an entity's `metadata` share it, as
  do one instant written in two time zones. Changing one to another does not
  move the revision. Nothing resolution reads can differ between them, but
  your own code reading `metadata` by type could.
- **The library release.** Equal answers mean equal content under one release
  of this library; another release may encode differently and move every
  revision. That reports a change that did not happen.
- **Alias order.** Reordering an entity's aliases moves the revision, though
  it resolves the same. That errs the same way.

The digest is `fingerprint_value` from `dataknobs_common`, the encoding
`StructuredConfig.fingerprint()` keys configs with, applied to any value. It
is there to call if you need a content digest of your own.

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

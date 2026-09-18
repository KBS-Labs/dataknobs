# Ontology Registry

An ontology in `dataknobs-common` is a **value**: it holds sources rather than
entities, owns no lifecycle, and is safe to hold and pass without anyone having
to remember to close it. Its two module-level doors — `load_ontology` and
`async_load_ontology` — therefore refuse a *live* source kind, and the refusal
is about ownership rather than about what is installed:

```
source 'products' declares kind 'record', which is a live source: it must be
opened, and closed again, and a module-level loader owns no lifecycle to do
that with. Load this ontology through OntologyRegistry, which does.
```

`OntologyRegistry` is the object that does. It lives in `dataknobs-data`
because that is where the handles are.

## The shape of it

```python
from dataknobs_config import EnvironmentAwareConfig
from dataknobs_data.ontology import OntologyRegistry

cfg = EnvironmentAwareConfig.load_app("catalog")
registry = await OntologyRegistry.from_config_async(
    cfg.resolve_for_build("ontology")
)

onto = registry.get("catalog")
entity = await onto.entity(onto.localize("catalog:sku-4471"))
assert entity is not None
entity.name                                 # "Beagle"

await registry.close()
```

Two things are worth reading off that. `registry.get(...)` hands back an
**`AsyncOntology`** — the lazy twin, because a vocabulary over a table cannot
answer without reading one — so `entity()` is awaited while `name` stays a
field. And `onto.localize(...)` is what turns a qualified id into the local key
`entity()` takes; the qualified id itself answers `None` rather than raising,
which is the one mistake worth making once.

## The two config forms, and which door takes which

A constructor takes the resolved form; a registry stores the portable one and
resolves it.

| Door | Takes | Who resolves |
|---|---|---|
| `OntologyRegistry.from_config_async(config, **components)` | the **resolved** form — `cfg.resolve_for_build("ontology")` | you, before the call |
| `await registry.load(config)` | the **portable** form — `$resource` intact, `${VAR}` unexpanded | the registry, at load |
| `OntologyRegistry.get_portable_config(cfg)` | an `EnvironmentAwareConfig`, or a mapping | nobody; it produces the storable form |
| `OntologyRegistry.from_components(config=..., database=...)` | handles already built | nothing to resolve |

The portable form is what a deployment *stores*, so it is the form a registry
reads back:

```python
registry = OntologyRegistry(environment="production")

stored = OntologyRegistry.get_portable_config(cfg)   # into your backend
onto = await registry.load(stored)                   # and back out of it
```

`load()` reads the ontology out of a document that carries other sections, and
uses a document that *is* the section as it stands — so both shapes you might
be holding resolve to the same vocabulary. Pass `config_key=` to name a
different section.

`from_components` is the door for a caller with no environment and no file — a
test, or an application that resolved its own handles:

```python
from dataknobs_common.ontology import OntologyConfig
from dataknobs_data.backends.memory import AsyncMemoryDatabase

registry = OntologyRegistry.from_components(
    config=OntologyConfig(id="catalog", sources=[...]),
    database=AsyncMemoryDatabase(),
)
onto = await registry.load()          # the document it was constructed with
```

An injected handle is used for every source in the document, and its
`database:` block is not resolved: a handle already built needs no name.

## A missing resource raises

```python
OntologyRegistry(environment="production", strict_resources=True)   # the default
```

| value | what a `$resource` this environment does not define does |
|---|---|
| `True` — **the default** | raises, naming the resource, the environment and the config path |
| `False` | degrades to the reference's inline defaults, with a warning |
| `None` | hands the level back to the environment's own `strict_resources` setting, then to `False` |

A reference's own `$required` overrides all three.

Strict by default because of what leniency produces over *this* grammar. A
`$resource` block is marker keys and nothing else, so it carries no inline
defaults to degrade to: a lenient resolution resolves it to `{}` and you get an
entity source built over an empty config, in an environment that was merely
missing a definition, announced by one `WARNING` at boot. Two shipped ways to
find that before it fires — `resolve_for_build(strict_resources=True)` as a
startup preflight, and `find_unresolved_resources` for every failure in one
pass rather than the first one.

## The binding

```yaml
ontology:
  id: catalog
  entity_types:
    - id: Product
  sources:
    - id: products
      kind: record
      database:
        $resource: catalog
        type: databases
      entity_projection:
        table: products
        id: sku
        name: title
        type: {const: Product}
        aliases: {column: alt_names, split: ","}
        description: blurb
        metadata: {columns: [owner_team]}
        surface_forms:
          table: product_forms
          form: folded_form
          entity: sku
      schema:
        - {name: sku, type: string}
        - {name: title, type: string}
        - {name: alt_names, type: string}
        - {name: blurb, type: string}
        - {name: owner_team, type: string}
        - {name: folded_form, type: string}
```

**The projection is data, never a callable.** That is what keeps the
column↦field map invertible, which is what a write path would one day rest on.
Every bare scalar names a column; the braced forms are the ones that do
something else — `{const: ...}` is a literal, `{column: ..., split: ...}` is a
delimited list, `{columns: [...]}` gathers metadata.

A column may be a **dotted path** into a JSON value (`name: payload.legal_name`),
and exactly one thing is promised about that: the *root segment* is validated at
load and the path within the value is not. A declared `payload` proves the
column exists and can say nothing about `legal_name` inside it, so a projection
reaching into a blob fails on first read.

### `schema:` is required

No database in this package exposes schema introspection, and a vocabulary over
foreign data is the case least likely to have declared one elsewhere — so a
binding declares the columns it projects or it is rejected at load, naming the
binding. A projection naming a column that declaration lacks is rejected naming
the column. Nothing here interpolates a name into a query: `Filter` is the only
path.

One `schema:` covers **both** of the binding's tables. `FieldSchema` carries no
table, so a name declared once is checked once and both uses are checked against
it — which is a naming constraint on your tables rather than a hole in the
check.

### `surface_forms:` is required under an `exact` rung

`by_surface_form` matches a form a person typed against forms **the source**
folded. A live table holds the form as it was written, and the fold the contract
names is `str.casefold` — full Unicode case folding, which no engine performs at
query time. (`'Straße'.casefold()` is `'strasse'`; every engine primitive is
simple lowercasing and leaves the ß alone.) So the fold happens before the row
is written or it does not happen at all, and the binding says where those rows
live:

```yaml
surface_forms:
  table: product_forms      # one row per (folded form, entity)
  form: folded_form         # holds normalizer(form)
  entity: sku               # -> this projection's `id:`
```

A table rather than a column, because one folded form reaches many entities and
one entity is reached by many forms — an entity's id, its name and every one of
its aliases all fold into it.

A binding whose document declares an `exact` rung and whose projection declares
no `surface_forms:` is **rejected at load**, naming the key. A binding with no
exact rung over it needs none and loads unchanged.

A source with no declared lookup **refuses** the call rather than answering over
the unfolded column:

```python
onto.describes[0].capabilities        # no Capability.SURFACE_FORM_LOOKUP
await onto.by_surface_form("beagle")  # CapabilityNotSupportedError
```

`frozenset()` already means *ran and matched nothing*, and a resolution cascade
falls through to a guessing rung on exactly that reading. A source answering the
unfolded column would return it for every query whose case differs by one
letter, and the cascade would report a vocabulary gap that is not there.

The fold is injectable, and it is one callable for both halves:

```python
RecordEntitySource(db, projection, source_id="products", normalizer=my_fold)
```

## What it refuses

| Refused | Because |
|---|---|
| a source kind nothing binds | computed from the declared kind alone, so one config means one thing whether or not some other package is importable |
| a document declaring both `entities:` and a live source | reads across both need a router that does not exist yet, and letting one win would be a silent choice about which entities exist |
| more than one live source | the same router |
| an ontology id already loaded | the registry instance is the unit of sharing; pass `replace=True`, or use a registry per tenant |
| `type:` naming a column | a source over a type column cannot yet report that its declared types are unenumerable, and reporting an empty set instead would read as a complete enumeration |

## Events

Loading, unloading and rebuilding are announced on an injected or configured
`EventBus`, as a **topic and a type**. Nothing is added to `EventType`.

| What happened | Topic | Type |
|---|---|---|
| loaded | `ontology:{id}` | `CREATED` |
| unloaded | `ontology:{id}` | `DELETED` |
| rebuilt | `taxonomy:{id}` | `UPDATED` |

```python
from dataknobs_common.events import InMemoryEventBus

bus = InMemoryEventBus()
await bus.connect()
registry = OntologyRegistry.from_components(config=..., event_bus=bus)
```

An unload carries the departing ids in one of two forms, decided by the sources
rather than by a size threshold: the **id set** where every bound source is
authored, and the `{ontology_id}:` **prefix** as soon as one is live — because
building the set over a live table is a scan on the way out.

A rebuild carries **three** sets — `gone`, `arrived`, and `renamed`, the last
being what changed name while keeping its id. A two-set delta reports a rename
as no change at all, which is why an empty payload here can be read as a
genuine no-op. The delta is over the *declared* population: a live binding's
rows change underneath the registry with no reload at all, which is what *live*
means.

## Teardown

```python
await registry.close()
```

`close()` releases every handle the registry **opened** and leaves every handle
it was **handed**. Ownership is recorded when a handle is acquired rather than
recomputed at teardown, because a registry that resolved one reference and was
given another has two handles with two different owners. The cascade is
error-isolated, so one failing handle does not abort the rest.

**`close()` does not unload, and `unload()` does not close.** The ids stay
listed after a close and the values stay reachable; what is gone is the ability
to read through them. And a value `get()` handed back outlives its entry — this
object cannot know whether you still hold one — so releasing a handle is your
decision, spelled `close()`.

```python
await registry.unload("catalog")   # the events, and the id leaves this registry
await registry.close()             # the handles
```

## Not yet here

`registry.index(id)` and `registry.resolver(id)` are declared and answer `None`
for every ontology this version builds. Absence is a configuration answer rather
than an error — an ontology that declared no `index:` section answers `None` for
good, and a registry with no index builder answers the same thing from the other
side. The `resolver:` section *is* read at load, which is where the
`surface_forms:` refusal above fires.

Write-through is not here either: `RecordEntitySource` declares no write
capability, which is a property it states rather than a gap it hides.

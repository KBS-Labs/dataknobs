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

Where a projection's `surface_forms:` names a **different table** than its
entity rows, one handle may not be enough — and whether it is enough is the
backend's answer, not the registry's. On `memory`, `file`, `s3` and
`elasticsearch` a handle *is* the store, so both kinds of row live in it and
one handle serves; on `sqlite`, `postgres` and `duckdb` a handle addresses one
table, and the second one is injected beside the first:

```python
registry = OntologyRegistry.from_components(
    config=OntologyConfig(id="catalog", sources=[...]),
    database=entities_handle,             # the projection's `table:`
    forms_database=forms_handle,          # its `surface_forms.table:`
)
```

Omitting it where it is needed is **refused at load**, naming both tables and
the handle's class. That refusal is there because nothing downstream could
notice: the entity reads would be correct, `describe()` would still advertise
`SURFACE_FORM_LOOKUP`, and `by_surface_form` would answer `frozenset()` — which
means *ran and matched nothing*, not *could not be served*. The configured door
opens both handles itself and needs no equivalent.

### The registry's own settings travel on any door

`environment`, `strict_resources`, `config_key` and `normalizer` are
properties of a *registry* rather than of a document, so none of them is a
field on `OntologyConfig`. Write them on whichever door you are already
using — the constructor names them, and the three published doors carry them
through the same keyword channel their collaborators arrive on:

```python
registry = await OntologyRegistry.from_config_async(
    cfg.resolve_for_build("ontology"),
    environment="production",      # a setting
    database=my_handle,            # a collaborator
)
```

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

**`type: {const: ...}` names a type the document declares.** It is the one
key of the projection that is a reference into another section rather than
into the table, and it is checked at load like the references the core
refuses: a `const:` naming no `entity_types:` row would type *every row of
the table* as something the vocabulary does not hold, so an index enumerating
by type would find none of them. A document that declares no `entity_types:`
at all is unaffected — an empty section is no schema rather than an empty one.

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

On `sqlite`, `postgres` and `duckdb` the two tables get a handle each, because
those backends declare a `table` on their config. On `memory`, `file`, `s3` and
`elasticsearch` the handle *is* the store, so both kinds of row live in one and
the form column is what tells them apart — `by_surface_form` filters on it and
reaches only form rows, and the entity-side reads exclude it and reach only
entity rows. An entity row must therefore not carry the form column, which is
the same thing the lookup direction already required.

A binding whose document declares a rung that *reads* folded forms — `exact`,
`scan` or `lexical` — and whose projection declares no `surface_forms:` is
**rejected at load**, naming the key and the rungs. Which kinds those are is
read from what each kind declares about itself, so a rung added later, or one a
consumer registers, is covered without an edit to this package.

A document declaring no `resolver:` section has written no rungs and loads
unchanged — but silence is the *default* composition, two of whose three rungs
read the member, so a cascade built over such a binding is refused when the
rungs are constructed rather than here. A binding loaded only to read entities
from is never refused for a cascade nobody builds.

Declaring the lookup is what `exact` and `scan` need, and it is **not** enough
for `lexical`. That rung scores the query against *every* form the vocabulary
declares — `list(catalogue.surface_forms())`, on each call — so it needs a
source that can enumerate its forms rather than one that can look a form up. A
`kind: record` binding is the second kind and not the first: a live table has
no bounded enumeration to offer, and pulling one would read the form table into
memory per query. `LexicalSignal` and its async twin therefore require a
surface-form *catalogue* — `AsyncSurfaceFormCatalog` on the flavour a registry
builds — and refuse anything else when the rung is **constructed**, whatever
the projection declares. The refusal names the source and the protocol.

Over a live binding, `exact` is the rung that matches a form exactly and `scan`
the one that finds it inside a longer utterance; near-spelling over a table is
a different mechanism and is not this one.

### A `scan` rung needs a bound the table cannot supply

A scanning rung probes every contiguous window of a query, and takes how wide a
window may be from the vocabulary's longest declared form. An authored index
counts its own keys and answers. A live table cannot be counted — and a count
taken at load is stale the moment a row is written — so the binding declares it:

```yaml
surface_forms:
  table: product_forms
  form: folded_form
  entity: sku
  longest_form_tokens: 4      # the longest form in this table is four tokens
```

Without it, the scan enumerates every window: *n(n+1)/2* probes for *n* tokens,
and on a live binding each probe is a database round trip — 1,275 reads for a
fifty-token utterance, 20,100 for a two-hundred-token paste. So a document
declaring a `scan` rung over a binding that declares no bound is **rejected at
load**, naming the key.

The number is not invented for you. A default would make a form longer than it
silently unfindable, which trades a cost problem for a correctness one; over an
in-memory vocabulary the same enumeration is a dictionary lookup per probe and
is left alone, which is why the refusal is here rather than on the rung. Which
kinds are bounded this way is read from what each kind declares about itself, as
with the lookup rule above.

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

The fold is injectable, and it is one callable for both halves — the table
you folded and the query folded against it:

```python
RecordEntitySource(db, projection, source_id="products", normalizer=my_fold)
```

Hand it to the registry instead when the registry is what builds the source,
and it reaches every source that registry binds — the live one and an authored
vocabulary alike:

```python
registry = await OntologyRegistry.from_config_async(resolved, normalizer=my_fold)
```

### `by_type` is a scan, and it is streamed

A projection declares one constant `type:`, so `by_type` over a live binding is
*every row* for that id — and there is no `limit:` that would make it something
else, because the answer is every id. What it costs is therefore one pass over
the table, on every call.

What it does **not** cost is the table in memory. The rows are read through
`stream_read`, so a batch of `Record`s is alive at a time rather than all of
them — which on the backends that page for real (Postgres by cursor; DuckDB,
SQLite and Elasticsearch by batch) is the difference between a bounded read and
one whose footprint is the table's. The set of ids it returns is still the whole
population, so a caller with a million-row binding gets a million-element set;
if that is the wrong shape, the fix is a type column on the table, which is also
what would let `declares` say it cannot enumerate.

Over a **shared** store the scan narrows to the entity rows, and that read
streams too. Its cost is bounded by the same filter: it reads the entity rows,
not the table.

That branch used to take `search()` instead, because `stream_read` and `search`
are separate implementations on every backend and they did not agree —
Postgres's `stream_read` open-coded its WHERE clause and silently dropped non-EQ
filters, which is what this narrowing is. With that fixed, keeping the branch on
`search()` would now *lose rows* rather than protect them: an unbounded `search`
is the read a backend is free to cap, and Elasticsearch caps one at
`size=10000`. Because Elasticsearch declares `index` rather than `table` it is
always a shared store, so a binding of more than ten thousand rows with a
declared `surface_forms:` was answering with ten thousand ids and reporting
nothing. Its streaming door goes through the scroll API, which no such cap
reaches.

### One handle per table, opened and connected by the registry

Whether a binding naming two tables needs two handles is the **backend's**
answer, not the registry's, and it is asked rather than assumed: the three SQL
backends declare a `table` on their config, so `products` and `product_forms`
are two handles there; `memory`, `file`, `s3` and `elasticsearch` declare none,
because for them the handle *is* the store and rows of both kinds live in it.
Both arrangements work — what tells the two kinds of row apart in a shared
store is the surface-form column, in both directions.

A handle the registry opens it also **connects**, because it opened it: every
backend but `memory` and `file` raises *Database not connected* on its first
query. A handle injected through `from_components` is connected by whoever
handed it over.
One handle per distinct resolved block and table, built once however many
bindings name it, and one at a time — so two concurrent loads over one
`$resource` share a connection rather than opening two. A reload against an
**unchanged** environment resolves to the same block, so it reuses what is
open and a reload loop accumulates nothing.

Against a **changed** one it does accumulate, deliberately. A reload resolving
to a different block is a different handle, and the superseded one stays open
until `close()` — releasing it at `replace=True` would release a handle that a
vocabulary `get()` handed out may still be reading through, which is the same
reason `unload()` releases nothing. A long-lived loop reloading against an
environment that keeps moving therefore grows one handle per distinct
resolution; the way to bound it is to close the registry, not to reload it
forever.

### Two bindings may not share a store they cannot be told apart in

`table:` narrows nothing on `memory`, `file`, `s3` and `elasticsearch`. It
survives as the discriminator that decides whether a *second handle* is opened
— which those four do not need — and as a `describe()` field. No query carries
it. So two ontologies in one registry, both binding one `$resource` there and
projecting two different tables, get **one store and no separation**:

```yaml
# ontology `catalog-products`        # ontology `catalog-suppliers`
database: {$resource: catalog}       database: {$resource: catalog}
entity_projection:                   entity_projection:
  table: products                      table: suppliers
```

Both read every row. `by_type("Product")` answers with the supplier ids too,
`get()` on a supplier id answers with that row projected under the wrong type
and name, and where both declare a `surface_forms:` lookup, `by_surface_form`
matches the other binding's form rows as well — three wrong answers, none of
them an error.

**The second load is refused**, naming both bindings, both sets of tables, and
the ontology already holding the store. A `$resource` of its own is the remedy,
or a backend that addresses a table — on `sqlite`, `postgres` and `duckdb` the
registry opens a handle per table and the same two documents are served
correctly, which is what makes this a property of the backend rather than a
limit on how many ontologies a registry may hold.

Two bindings declaring the **same** tables are left alone: the same rows read
as a catalogue entry by one document and as something else by the other is a
decision, and nothing here can tell it apart from a mistake. What is refused is
narrower — two bindings that named different tables and were handed one store.

This applies to the injected door for a reason of its own, and reaches further
there. `from_components` holds one handle for every document the registry
loads, so two documents through it share a store even where the handle is
`sqlite` — a table-addressed handle is bound to the *one* table it was built
for, so two bindings naming two tables through it are further wrong, not less.
The question is therefore asked of the handle both doors end up holding rather
than of the `database:` block only one of them has.

The separation that would lift this needs a column on your own table saying
which binding a row belongs to. A `kind: record` binding only ever **reads**,
so that column is yours to add rather than the registry's to write, and there
is no form of `table:` that could stand in for one.

## A taxonomy over a parent column

Rows that carry their parent's key are a hierarchy already. A `taxonomies:`
row declaring `kind: column` binds one, and what comes back is an
`AsyncTaxonomy` like any other — the walk is the same walk, and only what is
underneath it differs:

```yaml
ontology:
  id: catalog
  sources:
    - id: products
      kind: record
      # ...as above, and its `schema:` declares `parent_sku`
  taxonomies:
    - id: categories
      kind: column
      source: products          # a source THIS document declares
      parent_key: parent_sku    # the column holding the parent's key
      relation: parent          # required — what these edges MEAN
```

```python
onto = registry.get("catalog")
tax = onto.taxonomy("categories")

await tax.at(onto.localize("catalog:sku-4471")).ancestors()   # the chain, root last
await tax.structure.roots()                                   # the rows nothing places
```

`relation:` is required here as everywhere, and a column axis satisfies it by
naming what its edges *mean* rather than by naming a set of assertions to read.
Nothing constructs an `Assertion` for a row: the edges are two columns, and the
axis is a different backing under one taxonomy rather than a second kind of
taxonomy.

**So `parent_edges()` on this axis is `()` unless the document also asserts
that edge, and the member that says which is `has_edge_annotations()`.** Not
`assertions is None` — a registry builds an assertion source for *every*
vocabulary it loads, so a column axis's source is empty rather than absent,
and the field answers `False` for an annotated axis and an unannotatable one
alike. Measured over one five-row tree on both backings, `assertions is None`
is `False` both times while the edge read answers 0 and 1. The distinction
matters to any consumer who reads the walk and
`ontology.assertions.find(subject=...)` together, which the anchored view's own
published call site does: over a live binding the second half answers nothing,
with no error between the two answers.

A column axis is not, however, one on which nothing *can* be written. Its
edges come from the table, and an assertion under the axis's relation that
lands on one of them annotates it — measured, `parent_edges("mid")` answers 1
for a document carrying both the parent column and `mid --parent--> root`. So
`has_edge_annotations()` is not *is this axis column-backed*, and it is not
*does this vocabulary hold the relation* either: two axes may share a relation,
and a stated negation is not an annotation. It asks whether an **asserted**
edge under this axis's relation lands on an edge of *this* axis, which is what
`parent_edges()` reads, so the two agree by construction.

**There is no `child:`.** The child column is the source's own `id:`, which is
what keeps the axis and the entity source keyed alike by construction — the ids
a walk answers with are the ids `entity()` takes. A `child:` key would be the
first place two id spaces could diverge.

**`parent_key:` is checked against the same `schema:` the projection is.** One
more column over one declaration, checked at load and named on a refusal, for
the reason every other column name here is: a name from a config file reaches a
query builder and nothing interpolates it.

### A node is in the axis if an edge names it

Not *every row of the table*. That is the hierarchy protocol's own rule —
*"a node absent from every edge of this axis is not in this hierarchy at all,
so `roots()` equals a type's membership only by coincidence"* — and it is what
makes this axis answer identically to the same tree authored as `parent`
assertions. A row whose parent column is null **and** that nothing names as a
parent is not in the axis: `contains()` is `False`, `roots()` omits it, and a
cursor over it reports `exists()` `False` and `ancestors()` `()`.

An edge is a row with **both** ends, spelled as two `EXISTS` filters, which
mean *is not null* on every backend here.

Over a **shared store** — `memory`, `file`, `s3`, `elasticsearch`, where the
handle *is* the store and form rows sit beside entity rows — the axis is
narrowed by the same filter the entity reads are narrowed by, taken from the
binding rather than derived again. A form row that carried a parent column
would otherwise be an edge of the axis while being a row the entity source
refuses to project, so `parents()` and `entity()` would disagree about one id.
Nothing declares that a side table carries no parent column, and a
denormalised one generated by a join carries whatever it was joined from.

Referential integrity is not the axis's subject. A parent column naming a key
no row carries places a node `entity()` will not find — exactly as an assertion
may name an entity no `entities:` row declares.

!!! warning "A **blank** parent cell is an edge, and a blank is not a null"

    *Is not null* is the rule, not *is falsy* — so an empty string in the
    parent column is an edge to a node whose id is `""`. Nothing raises: the
    tree still walks and a roll-up still places every node. What is wrong is
    that `roots()` answers with the absence of a key read as a key, and every
    ancestor chain gains an element `entity()` returns `None` for.

    This is not the referential-integrity case above. That one is somebody
    writing an id and the id being wrong. **This is a blank read as a key,
    which nobody wrote** — and it is the single most common origin of a
    `parent_id` column: a CSV export writes `""` for an absent value, and so
    does a form that stores a blank rather than a null.

    The rule is not bent to accommodate it, and the file's own history is why.
    `EXISTS` on the parent column was relied on once before to exclude a class
    of row, which was *"a property of the data rather than of the declaration,
    and nothing enforces it"*; the repair was a narrowing filter taken from the
    binding, **not** a change to what `EXISTS` means. Treating `""` as null
    would be that bend taken the second time, and it would quietly refuse a
    legal key.

    So the repair is at the source: make the column nullable and write `NULL`
    for a root, or narrow the binding so the blank rows are not projected.
    Until then, a root will show as a child of `""`.

**The ids are read as strings**, which is the projection's own rule rather than
this axis's: `RecordEntitySource` projects a row's id column through `str()`
too, so the two agree. An integer `id:`/`parent_key:` pair therefore walks in
the string space — `5` answers as `"5"` — and a consumer holding integer keys
converts at the boundary, once, rather than meeting the question per member.

### It reads in one query per level, and opens nothing

`ColumnHierarchy` carries the two optional hierarchy protocols, not just the
four singular members, and a row-backed axis is the case those protocols were
written for. A frontier costs one query rather than one per node, and
`parent_edges()` answers the whole axis in one read — which is what lets
`materialization.structure: materialized` take a copy that is both one query
and *complete*. A copy built by descending from the roots cannot reach a cyclic
component with nothing above it, and would then refuse an anchor the live axis
accepts. Nothing stops a `parent_id` column from carrying a cycle.

Binding an axis opens **no handle**. It reads the source's table, and a handle
is built once per resolved block and table, so the axis is handed the one the
source already holds: `close()`'s cascade is unchanged and the axis cannot
outlive the source beside it.

### The class is public

```python
from dataknobs_data.ontology import ColumnHierarchy

axis = ColumnHierarchy(db, "products", "sku", "parent_sku")
await axis.parent_edges()
```

Configuration is the ordinary way to get one, and the constructor is there for
a caller who already holds a handle — the four arguments are the handle, the
table it addresses, the column holding each row's own key, and the column
holding its parent's. A fifth, `narrowing=`, takes the filters every read of
this axis is additionally bounded by; the registry fills it with the binding's
own, and a caller over a store holding nothing else leaves it empty.

## What it refuses

| Refused | Because |
|---|---|
| a source kind nothing binds | computed from the declared kind alone, so one config means one thing whether or not some other package is importable |
| a document declaring both `entities:` and a live source | reads across both need a router that does not exist yet, and letting one win would be a silent choice about which entities exist |
| more than one live source | the same router |
| an ontology id already loaded | the registry instance is the unit of sharing; pass `replace=True`, or use a registry per tenant |
| a second binding over a store another ontology already holds, naming different tables | one handle serves both and no read narrows to a binding's table there, so each would answer with the other's rows |
| `type:` naming a column | a source over a type column cannot yet report that its declared types are unenumerable, and reporting an empty set instead would read as a complete enumeration |
| an axis kind nothing binds | computed from the declared kind alone, for the source kinds' reason. An axis over this document's own assertions declares no `kind:` at all |
| a column axis naming a source the document does not declare | an axis reads one document's table, and a source id is resolved in the document that wrote it |
| a column axis over an **authored** source | there is no table under it, so there is no column to read — refused rather than downgraded to the assertion read, which would be the silent substitution one step later |
| a `parent_key:` the binding's `schema:` does not declare | the projection's rule, over one more column of the same declaration |

## Events

Loading, unloading and rebuilding are announced on an injected or configured
`EventBus`, as a **topic and a type**. Nothing is added to `EventType`.

| What happened | Topic | Type |
|---|---|---|
| loaded | `ontology:{id}` | `CREATED` |
| unloaded | `ontology:{id}` | `DELETED` |
| rebuilt | `ontology:{id}` | `UPDATED` |
| rebuilt, one axis | `taxonomy:{id}` | `UPDATED` |

A bus is either injected or configured. Injected:

```python
from dataknobs_common.events import InMemoryEventBus

bus = InMemoryEventBus()
await bus.connect()
registry = OntologyRegistry.from_components(config=..., event_bus=bus)
```

Configured — an `event_bus:` block beside `id:` in the ontology document,
which the registry builds and closes because it built it:

```yaml
ontology:
  id: catalog
  event_bus:
    backend: memory
  sources: [...]
```

One level and one reader: the block sits in the ontology section, where
`index:` and `resolver:` sit, and both doors read it after resolution — so a
`${VAR}` in a connection string is expanded and a bus configured this way is
connected before the load it announces. An injected bus always wins, and is
never closed by the registry.

The block is built **off the event loop**, for the reason a handle is: every
backend factory imports its driver — `asyncpg`, `redis`, `aioboto3` — inside
the factory call, so that a base install pulls none of them, and an import is
disk I/O. A backend whose *construction* is genuinely asynchronous is built by
its owner and injected.

An unload carries the departing ids in one of two forms, decided by the sources
rather than by a size threshold: the **id set** where every bound source is
authored, and the `{ontology_id}:` **prefix** as soon as one is live — because
building the set over a live table is a scan on the way out.

A rebuild carries **three** sets — `gone`, `arrived`, and `renamed`, the last
being what changed name while keeping its id. A two-set delta reports a rename
as no change at all, which is why an empty payload here can be read as a
genuine no-op.

**A topic carries a delta over what that topic is about**, so a rebuild
publishes at two levels, and the two levels are over different populations
because they cost different things to enumerate. The ontology's own topic gets
the delta over its **declared** entities — a live binding's rows change
underneath the registry with no reload at all, and counting them on the way
past is a scan per type. Each axis topic gets the delta over *that axis's
nodes*, whatever backs them, because an axis answers its whole population in
one query. Neither level contains the other: an entity in no axis at all would
be announced nowhere, and an id an axis places that no `entities:` row names is
in no ontology-level set.

An axis's population is read **when the vocabulary is built** and compared
against the next build's. That is the only comparison a live axis can answer:
reading the outgoing axis at the rebuild asks the same rows the same question
twice and reports that nothing changed however much did.

**It costs one read of the table per column axis, per load**, held as a set of
ids until the next load replaces it — so an axis over a materialized structure
reads the table twice at one load, once for the snapshot and once for this. It
happens only where the registry holds a bus, since the population is a delta's
input and there is nobody to tell otherwise. That is the asymmetry with the
ontology level above: one read per axis is a cost worth paying for a delta that
is true, and a scan per declared type is not.

Where a population is **not known**, the payload carries
`axis_unenumerable: true` and **none of the three sets** — a subscriber reading
`payload["gone"]` gets a `KeyError` rather than an empty list, which is the
point: it says re-read the axis rather than saying nothing happened. The case
that reaches it is a registry that acquired its bus *between* two loads, the
first having recorded nothing; any document may declare an `event_bus:` block,
so any load may be the one that brings it. (A backing that cannot enumerate
itself would take the same form, since the enumerable protocol is opt-in by
member presence — but every backing a door files here answers `parent_edges`,
so nothing the registry builds reaches it that way.)

Every axis either document declares gets an event, in the order the new one
writes them and then the ones it no longer does. An axis a rebuild **removed**
reports its whole population `gone`, which is the strongest thing its
subscribers can be told: the axis they read is not there any more.

## Teardown

```python
async with await OntologyRegistry.from_config_async(resolved) as registry:
    ontology = await registry.load(document)
    ...
# every handle the registry opened is released here, on the way out of the
# block -- including the way out an exception takes
```

The block is `close()` exactly, so everything below is what it does. Entry
builds nothing and hands back the registry: a handle connects on entry to
`AsyncDatabase`'s block because a handle is the thing being opened, and a
registry opens handles at `load()`, when a document says which ones. Writing
the call yourself is equally supported and is what the examples above do:

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

**It is idempotent**: what the registry owned is closed and forgotten, what it
was handed is untouched and still recorded. So a second `close()` closes
nothing twice, and an `unload()` after a close still announces its departure
when the bus is an injected one — which is the case where a bus is still there
to hear it.

```python
await registry.unload("catalog")   # the events, and the id leaves this registry
await registry.close()             # the handles
```

## A semantic index over the vocabulary

An `index:` section binds the vocabulary to a vector store, and
`registry.index(id)` hands back what the load built.

```yaml
ontology:
  id: catalog
  version: "1.0"

  entity_types:
    - id: Product
    - id: Brand

  entities:
    - {id: sku-4471, type: Product, name: Acme Widget, description: a widget,
       aliases: [Widget, "ACME widget"]}
    - {id: acme, type: Brand, name: Acme Corp}

  index:
    store:
      $resource: catalog_index
      type: vector_stores
```

```python
registry = OntologyRegistry.from_components(
    config=OntologyConfig(**document), embedder=embedder
)
await registry.load()

index = registry.index("catalog")
written = await index.build()          # every entity, one row each
hits = await index.search("acme widget", k=5)

hits[0].record.id                      # 'catalog:sku-4471' -- qualified
hits[0].metadata["dk_ontology_id"]     # 'catalog'
registry.get("catalog").localize(hits[0].record.id)   # 'sku-4471'
```

**The enumeration is the vocabulary, not the schema.** Every entity the bound
source holds gets a row, under every type it actually holds — so a type
`entity_types:` declares and nothing is filed under contributes nothing and
raises nothing.

An entity whose type the schema never declared is *not* silently dropped — but
what happens to it depends on whether there is a schema at all. Where the
document declares `entity_types:`, such a type is **refused at load** and the
vocabulary does not come up; where it declares none, the by-reference regime is
the case this indexes rather than a document declaring nothing, so the entity is
indexed. An empty schema section is no schema, not an empty one.

**Every row carries a qualified id.** Qualification happens in the source; the
index stores what it is given and returns what it stored, and never rewrites an
id or infers a namespace. `localize` reads each one back.

### The embedder is injected, and a configured one is checked

The store the registry opens itself — `store:` resolves through the same
`$resource` mechanism a `database:` block uses, and the handle is released by
`close()` like any other. **The embedder it cannot open.** An `embedder:` block
resolves to a provider-and-model pair, and the only construct that accepts one
lives in `dataknobs-llm`, which depends on this package — so the edge runs the
wrong way for this package to build one. An embedder also has no `close()` at
all; the socket is one level below it, on the provider it holds, so a registry
that built one would claim a responsibility it could not discharge.

So it is handed in:

```python
from dataknobs_llm.llm.embedding import create_text_embedder

registry = await OntologyRegistry.from_config_async(
    resolved, embedder=await create_text_embedder(...)
)
```

`create_text_embedder` is `async def`, so the `await` is load-bearing: without
it the registry is handed a coroutine object where a `TextEmbedder` is
expected, and the failure arrives at the first `build()` rather than here.

and a document declaring an `embedder:` block with nothing injected is
**refused at load**, naming what to pass. Refused rather than dropped: a
section parsed into a field and discarded leaves a registry reporting success
while holding no store, no embedder and no index, and no error anywhere.

Declaring one **and** injecting one is a **claim the registry checks** — the
same shape `metric:` takes one key along, a value a document may state that is
compared rather than applied. The declared `model:` is compared against the
injected embedder's `model_id`, and a disagreement is refused at load naming
both sides. So a document naming `nomic-embed-text` beside a process injecting
MiniLM no longer indexes under MiniLM while the document says otherwise; it
does not load.

This was a refusal, and the comparison serves that refusal's own intent more
completely. The intent was to make the disagreement visible; a refusal also
left the document unable to state the model **at all**, which matters because
a `kind: semantic` rung's `threshold:` is a raw distance in one particular
model's geometry and had nowhere to name the model it was calibrated against.
The refusal's stated reason — *cannot build the declared one to compare* — is
about **building**, and the injected embedder already publishes `model_id`.

**The matching rule is two omissions, and either side may make either one.**
`LLMProviderEmbedder.model_id` is `provider:model` and an Ollama model name
carries its own `:tag`, so that embedder publishes
`ollama:nomic-embed-text:latest` where a document naturally writes
`nomic-embed-text`. The **provider** is a prefix and the **tag** is a suffix,
and requiring either of them of either side would refuse a correctly
configured deployment.

**`model_id` promises no format**, and the other two implementations in this
workspace do not use that one: the bots knowledge-base adapter publishes
`kb:<name>` and `DeterministicEmbedder` publishes a bare word. A consumer
implementation wrapping an Ollama client publishes `nomic-embed-text:latest`
— a tagged model with no provider, and the same shape as a provider and a
model. Nothing in the string tells the two apart, so the comparison reads the
published identity **both ways** and the document agrees when either reading
does. A provider **both** sides state must agree, because two providers
serving a same-named model are not the same weights and a threshold
calibrated against one means nothing against the other; a provider only the
document states contradicts nothing, because the embedder published none.

*May omit* is not *is always ignored*. Where **both** sides state a tag the
tags are compared, because `:v1.5` beside `:latest` is two sets of weights —
and a version bump is the likeliest single change to invalidate a `threshold:`
calibrated in one model's geometry, which is what this comparison is for. So
the one disagreement a both-sides strip cannot see is the one most worth
seeing.

| The document writes | Injected `model_id` | Verdict |
|---|---|---|
| `model: nomic-embed-text` | `ollama:nomic-embed-text:latest` | agrees — the document omits the tag |
| `model: nomic-embed-text:latest` | `ollama:nomic-embed-text` | agrees — the embedder omits it |
| `model: nomic-embed-text` | `nomic-embed-text:latest` | agrees — the embedder publishes no provider |
| whatever the embedder publishes, verbatim | that same `model_id` | agrees — always |
| `model: nomic-embed-text:v1.5` | `ollama:nomic-embed-text:v1.5` | agrees — both state it, and they match |
| `model: nomic-embed-text:v1.5` | `ollama:nomic-embed-text:latest` | **refused** — both state a tag and they differ |
| `provider: ollama`, `model: nomic-embed-text` | `ollama:nomic-embed-text:latest` | agrees |
| `provider: ollama`, `model: nomic-embed-text` | `nomic-embed-text:latest` | agrees — no published provider to contradict |
| `model: mxbai-embed-large` | `ollama:nomic-embed-text:latest` | **refused** — different model |
| `provider: openai`, `model: nomic-embed-text` | `ollama:nomic-embed-text` | **refused** — different provider |
| `provider: ollama` alone | anything | **refused** — a claim with no model in it cannot be checked |

One case the reading rule cannot judge, stated rather than left to be found:
a document naming the published id **minus its last colon-segment** agrees —
`model: ollama` against `ollama:nomic-embed-text`, where the whole-string
reading treats `nomic-embed-text` as an omitted tag. That document is wrong
and nothing here catches it, because catching it means assuming the format
`model_id` declines to promise. Two genuinely different model names still
disagree under every reading, which is what the comparison is for.

The last row is the `metric:` precedent again: a claim the registry cannot read
is not a claim it may pass over. An injected embedder publishing **no**
`model_id` is the one case that warns instead of raising — the document is not
what is wrong, `model_id` is optional on the protocol, and refusing would make
a legitimate embedder unusable with a legitimate document.

#### What the `embedder:` block reads

`model:` and `provider:`, **and a key that is not one of them is refused** —
`index:`'s own rule one level in, and it bites harder here because this block
is a *claim* rather than a spec. The registry cannot build an embedder and
does not try, so `dimensions: 256` or `api_base:` written here configures
nothing whatsoever. Reading two keys and passing over the rest is the same
silent drop the no-`model:` refusal above names as its own reason for
existing; and a block whose `model:` is visibly checked invites the reader to
assume the keys beside it are too.

The pair may be written flat or under **one** named nesting, `embedding:`,
which is how `dataknobs-llm` spells its own configuration:

```yaml
  index:
    embedder:                       # flat
      provider: ollama
      model: nomic-embed-text

  index:
    embedder:
      embedding:                    # or nested, and nothing beside it
        provider: ollama
        model: nomic-embed-text
```

One level or the other, not both. Splitting the pair —
`embedder: {provider: ollama, embedding: {model: nomic-embed-text}}` — is
refused, because only one level is read and the other half would be dropped
in silence. That nesting is also named rather than searched for: reading a
`model:` out of *whichever* sub-block happened to carry one made
`embedder: {retry: {model: ...}}` refuse a correctly configured deployment,
naming as the document's declared model one the document never declared.

`model_id` also survives wrapping: it is a `TextEmbedder` protocol member and
both shipped decorators forward it unchanged — the cache so a vector stored
through it carries the same staleness key, the bridge so a vector stored
through it is indistinguishable.

### What `index()` answers, and what `unload()` does to it

`registry.index(id)` **returns** what `load()` built and is not a second way to
build one — a caller who loaded no `index:` section gets `None` however many
stores they hold. `unload()` drops the index with the vocabulary's other
per-id state and releases no handle, for the reason it releases none of the
others; `reload()` rebuilds it by the path that built it, reusing the store
where the resolved block has not changed. `close()` is what releases the store.

**The metric is a claim, not a setting.** `index:` may name one, and it is
checked against the store's own rather than applied to it — the store resolves
its metric at construction and its search takes no such argument. Two spellings
of one metric agree (`l2` and `euclidean` are one family); a real disagreement
is refused at load, as is a spelling the library does not resolve. Naming none
checks nothing, which is the only safe default: defaulting to cosine would
refuse every euclidean store somebody configured on purpose.

### What the `index:` block reads

Six keys, and **a key that is not one of them is refused**. A block is
configuration a consumer wrote, and one the registry accepts while acting on it
in no way is the same silent drop the `embedder:` refusal above exists for —
`metirc: l2` would otherwise build an index with no metric check and report
success.

| Key | What it does |
|---|---|
| `store:` | Where the vectors go. No default; resolved through `$resource` and released by `close()`. |
| `embedder:` | A claim, checked against the injected embedder's `model_id` rather than built. Reads `model:` and `provider:`, and refuses anything else — see [what the block reads](#what-the-embedder-block-reads). Refused with nothing injected — see above. |
| `metric:` | A claim about the store, checked rather than applied. |
| `fields:` | Which entity fields compose the embedded text, in order. `name` and `description` are the two an entity carries free text in; anything else is refused at load. Defaults to `[name]`. |
| `join:` | What goes between two non-empty field values. Applied *between* them, so an entity carrying one of two produces no dangling separator. Defaults to ` -- `. |
| `aliases:` | `true` wraps the source so each surface form is indexed as its own item under the entity's id. See the warning below before using it. Checked for being a boolean, not for truthiness — `aliases: "no"` is truthy. |

```yaml
  index:
    store:
      $resource: catalog_index
      type: vector_stores
    fields: [name, description]
    join: " -- "
    aliases: true
```

!!! warning "`aliases: true` collapses in a store keyed on id"

    The decorator yields one item per surface form, all carrying the entity's
    id, and a vector store upserts on id conflict — so three forms leave **one**
    row, holding the last form rather than the entity's name. Which id the extra
    rows should take is not settled by any ruling this layer can read. Use it
    where the reader de-duplicates by id, or where the store is keyed on
    something else, until it is.

!!! note "A `fields:` the rows do not fill is reported at the build, not at load"

    `fields:` is checked at load for *naming* something an entity carries —
    `name` and `description`, and nothing else. It cannot be checked for being
    *filled*: a live binding's projection fills whatever columns the consumer's
    table has, so `fields: [description]` over a table without that column is a
    legitimate document right up until the rows arrive.

    The stream reports it instead. Where every item a build streamed composed
    empty text, `EntitySourceIndexSource` logs a warning naming the ontology,
    the count, the fields and the source. What that configuration produces is
    not an empty index — it is an index of rows equidistant from every query,
    which is worse, because an empty index has a report and this has an answer.

    **The report follows the stream's close rather than its end**, so a build
    that fails partway carries it too and the message says the stream did not
    finish. A binding filling the field on *some* rows is not reported: that is
    ordinary, and so is a vocabulary holding no entities at all.

**Nothing that can be refused before the store opens is refused after it.** The
unread-key check, both embedder refusals, the source the block configures and
the spelling of `metric:` are all settled first, so a document that cannot build
an index does not leave an opened store behind proving it tried. The one check
that cannot move is the metric *claim*, which is a comparison against the
store's own metric and so needs the store it is about.

Every refusal here raises `ValidationError` — including the two that used to
escape as bare `ValueError`s, a misspelled `metric:` and a metric the store is
not serving. What is being refused in each case is a document.

## Resolving against the vocabulary

A `resolver:` section builds the placement cascade, and `registry.resolver(id)`
hands back what the load built. A `kind: semantic` rung searches the index
this same document declared — it is the one rung a document cannot configure
from a value-level door, because it is constructed over a live index rather
than over the vocabulary alone.

**The block below opens one step earlier**, because a deployment holding more
than one vocabulary has to pick before it resolves. It loads two, asks which of
them a page of tagged results is about, and resolves against the answer.
`ontologies_in_play` is that step, and *Which vocabularies a page of results is
about* below the fence is where it is taught.

<!-- worked-call-site -->
```python
import asyncio
from typing import Any

from dataknobs_common.entity_resolution import ResolutionResult
from dataknobs_common.ontology import (
    NODE_ID_KEY,
    ONTOLOGY_ID_KEY,
    TAXONOMY_ID_KEY,
    OntologyConfig,
    read_node_tags_many,
)
from dataknobs_data.ontology import OntologyRegistry
from dataknobs_data.testing import DeterministicEmbedder

document = {
    "id": "catalog",
    "version": "1.0",
    "entity_types": [{"id": "Product"}, {"id": "Brand"}],
    "entities": [
        {
            "id": "sku-4471",
            "type": "Product",
            "name": "Acme Widget",
            "description": "a widget",
            "aliases": ["Widget"],
        },
        {"id": "sku-8802", "type": "Product", "name": "Bolt", "description": "a threaded fastener"},
        {
            "id": "acme",
            "type": "Brand",
            "name": "Acme Corp",
            "description": "the maker of the widget",
        },
    ],
    "index": {
        "store": {"backend": "memory", "dimensions": 32},
        "fields": ["name", "description"],
    },
    "resolver": {"rungs": [{"kind": "exact"}, {"kind": "semantic"}]},
}

# A second vocabulary, loaded into the same registry. One is the case where the
# question below has a single answer and nobody has to ask it -- the registry
# instance is the unit of sharing, so a deployment holding several holds them
# here, in one flat id space.
suppliers = {
    "id": "suppliers",
    "version": "1.0",
    "entity_types": [{"id": "Supplier"}],
    "entities": [{"id": "acme-co", "type": "Supplier", "name": "Acme Corporation"}],
}

# What your own retrieval handed back. This page wrote none of it: rows out of
# somebody else's store, carrying whatever tags were written on them. Row 1
# touched no vocabulary at all, which most of a corpus does.
rows: list[dict[str, Any]] = [
    {ONTOLOGY_ID_KEY: "catalog", TAXONOMY_ID_KEY: "kinds", NODE_ID_KEY: ["sku-4471"]},
    {"invoice_id": "2291"},
    {ONTOLOGY_ID_KEY: "suppliers", TAXONOMY_ID_KEY: "trade", NODE_ID_KEY: ["acme-co"]},
    {ONTOLOGY_ID_KEY: "catalog", TAXONOMY_ID_KEY: "kinds", NODE_ID_KEY: ["sku-8802"]},
]


async def main() -> ResolutionResult:
    registry = OntologyRegistry.from_components(
        config=OntologyConfig.from_dict(document),
        embedder=DeterministicEmbedder(dimensions=32),
    )
    await registry.load()
    await registry.load(suppliers)
    try:
        # Read the tags once and keep the reading. `.tags` is the answer and
        # `.malformed` is the report: a row whose writer wrote two of the three
        # tag keys is refused by name rather than passing as untagged, and
        # taking `.tags` inline off the call is how that report gets dropped.
        reading = read_node_tags_many(rows)
        assert not reading.malformed  # `reading.require_readable()` raises instead

        # Which of the vocabularies this registry holds is that page of results
        # even about? Ranked by how many rows named each, so the answer is also
        # a measure of how much of it, and the rows ride along so that picking
        # one costs no second pass over the corpus. The axis id on each row is
        # not read here -- rolling a corpus up onto one axis is what needs it.
        in_play = registry.ontologies_in_play(reading.tags)
        assert [(s.ontology_id, s.rows) for s in in_play] == [
            ("catalog", (0, 3)),
            ("suppliers", (2,)),
        ]
        about = in_play[0].ontology_id  # "catalog", on two rows against one

        # `load()` assembles the index; filling it is yours. Nothing here
        # re-embeds a vocabulary on every reload behind your back.
        index = registry.index(about)
        assert index is not None  # `None` where a document declares no `index:`
        await index.build()

        resolver = registry.resolver(about)
        assert resolver is not None  # and `None` where it declares none
        result = await resolver.resolve("Acme Widget", k=5)

        best = result.candidates[0]
        assert best.entity_id == "sku-4471"  # local, not "catalog:sku-4471"
        assert best.declared  # a form the vocabulary carries
        assert [e.signal for e in result.explain("sku-4471")] == ["exact", "semantic"]

        tail = result.candidates[-1]
        assert not tail.declared  # only the embedding proposed it
        assert [e.signal for e in tail.evidence] == ["semantic"]

        return result
    finally:
        await registry.close()


result = asyncio.run(main())
```

### Which vocabularies a page of results is about

**Which vocabularies a page of results is even about is a question this
registry answers, and the answer is a ranking.** `ontologies_in_play` takes the
tags a corpus carries — one row's per position, as `read_node_tags_many` reads
them — and reports one entry per vocabulary *this registry holds* that the rows
named, ranked by how many rows named each, ties in the order the corpus first
named them. The rows travel with each entry, positionally, so picking one and
going on to use its evidence costs no second pass over the corpus.

**It narrows and does not count.** The measure is `ontology_support` in
`dataknobs_common.ontology`, which counts over the tags alone and holds no
registry at all; this member filters that answer to the ids it carries.
Filtering a ranked tuple preserves both the ranking and the tie-break, so there
is no second ordering here to disagree with the first — and no second
implementation of the measure to drift from it. The roll-up guide in
`dataknobs-common` is the same operation one level down, over the nodes of one
axis rather than over vocabularies:
<https://kbs-labs.github.io/dataknobs/packages/common/roll-up/>.

**What it drops is recoverable and is not reported.** A tag naming a vocabulary
this registry never loaded is gone from the answer, and nothing says it was
there. The door back is one import: `ontology_support` over the same tags
answers every vocabulary the rows named, held or not, so the residue is one set
difference. `()` therefore has more than one producer — a corpus carrying no
tags, a corpus naming only vocabularies you do not hold, and a registry holding
nothing. The tags separate the first two, and they say nothing at all about the
third: `list_ids()` is what answers that one.

**Which matters because the answer is over what a registry has loaded, not over
what its configuration names.** `from_config` is the synchronous door and it
loads nothing — a registry built through it answers `()` for every corpus until
a `load()` has been awaited, even though its config named a vocabulary. Every
other member you would reach for that early is a coroutine, so the await is
forced; this one is not, which is exactly why the empty answer is worth one
`list_ids()` before you read anything into it.

### Filling the index is yours to call

**`registry.index(id).build()` is yours to call, and the cascade says so when
you have not.** A load assembles the index and returns; it does not embed the
vocabulary. The alternative costs a full re-embed on every `load(replace=True)`
over a store whose rows survived the reload — and the obvious remedy, building
only when the store counts zero, is worse: a vocabulary that *changed* would be
skipped and the rung would answer from stale vectors. Which of those you want
is a deployment's choice, so it stays the caller's line.

Until you make it the rung is searching an empty store, and every candidate
comes from the declared rungs alone — an answer that looks ordinary. So the
rung counts rows the first time it finds nothing and logs once, naming the
ontology and the call above. It distinguishes two states, because the remedy
differs: **no rows at all** is the first-run case above, and **rows that are
all some other vocabulary's** is a store two documents share, where this
ontology's index was never built into it.

The check runs **once per rung**, whatever it finds. It used to cache only the
empty verdict, so a populated store was re-counted on every empty answer — and
`count()` is a `SELECT COUNT(*)` on pgvector, so any threshold a deployment
sets bought a sequential scan per query.

**The candidates are in the ontology's id space.** Every row the index holds is
a *qualified* id and a cascade's entities are local ones, so the rung localizes
on the way out. Without that, one entity two rungs both reached comes back
twice under two keys, `explain` finds each under a different one, and nothing
downstream can tell that they are one entity.

**A scoped resolve keeps this rung.** `within=` is rendered as a metadata
filter, and no row an ontology index writes carries a scope axis — a store
fails a row that is missing the key a filter names, so a rung that forwarded it
would answer nothing under every scope, which reads as *not in the corpus*. It
declares that it does not narrow, the cascade offers it no filter, and the
cascade rules on what comes back.

**It does send one filter of its own, and that is a different question.** Every
read this rung makes is scoped to `dk_ontology_id` — the key the index source
writes on every row, and the one a source's `declares()` is documented as the
translation target for. Without it a store two documents share answers rows
this vocabulary never declared, whose ids the ontology then refuses: a rung
that constructs cleanly and fails on *every* hit it finds. The constructor's
guard cannot reach that, because asking whether an index holds this ontology's
ids is a question about membership and this is a question about exclusivity.

Sharing is the easy case to reach, not the exotic one. The registry caches a
vector store on its resolved `store:` block alone, so two documents both
writing `{backend: memory, dimensions: 384}` get one store, and a `table:` or a
collection two deployments name is shared by construction.

### `threshold:` is a number about a model, not about a vocabulary

A `kind: semantic` rung takes a `threshold:` and drops every hit scoring below
it. **The number means nothing on its own.** It is a raw distance in one
particular model's geometry, and the same document under a different embedder
is a different filter.

Measured over a thirty-one-row corpus with the **identical document** either
side: `threshold: 0.4` fires on **2 of 31** rows under the fence embedder every
worked example here stands in with, and **30 of 31** under `nomic-embed-text`.
One declared number, two behaviours, no error either way. A fence embedder's
geometry is not a weak version of a model's — it is unrelated to it, so a
threshold tuned against one carries no information about the other.

Two things follow:

- **Say which model the number is for.** `index.embedder:` is where, and it is
  checked against the injected embedder rather than ignored — see *[The embedder is injected, and a configured one is
  checked](#the-embedder-is-injected-and-a-configured-one-is-checked)* above. A
  threshold and the model it means something against used to sit in different
  places with nothing pairing them.
- **Do not reach for a threshold to separate good hits from bad ones.** On that
  same corpus the best precision **any** threshold reaches is `0.167`, at
  `>= 0.55`, and it costs two of the three right answers. Filtering to the
  candidates that carry a **declared** form restores `1.000` precision and
  `1.000` recall, under both embedders. The discriminator that works already
  ships, and it is `EvidenceKind.DECLARED` — not a number.

Use `threshold:` for what it is: a floor that stops an unthresholded *k*-nearest
search from returning the whole corpus when the query is about nothing in the
vocabulary. `None` — the default — keeps every hit the store returns, which is
what an unthresholded cosine search means.

### What the `resolver:` block reads

One key, and **a key that is not it is refused** — for the `index:` block's
reason, sharpened: a section with no `rungs:` is read one layer down as a
composition of *nothing*, so `rung:` would build a cascade that matches nothing
and report success.

| Key | What it does |
|---|---|
| `rungs:` | The composition, in order. Each is a `{kind: ...}` mapping plus whatever that kind takes. |

Absence and emptiness are different answers, and **the line is drawn at the
section rather than at the list** — which is where this block differs from
`index:`, whose empty `{}` means *no index*. No `resolver:` section at all is
silence, and `registry.resolver(id)` answers `None` — compose your own. An
explicit `rungs: []` is a composition somebody chose, and it resolves nothing,
because the composition *is* the policy. An empty `resolver: {}` is read the
same way as `rungs: []`: the section exists, so `registry.resolver(id)` answers
a cascade, and that cascade holds no rungs.

Every way a composition can fail to build is refused at load as a
`ValidationError`, naming the offending entry's position:

| The document writes | What is wrong |
|---|---|
| `rungs: [exact]` | an entry that is not a rung's configuration |
| `rungs: [{threshold: 0.5}]` | an entry naming no `kind:` |
| `rungs: [{kind: sematnic}]` | a kind nothing registers |
| `rungs: [{kind: semantic}]`, no `index:` | a rung whose handle this document never declared |

The first three are refused **before** anything is opened, which is the
`index:` block's own rule applied to its sibling — a document that cannot build
a cascade must not leave a vector store behind proving it tried, and
`from_config_async` never returns the object whose `close()` would release one.
They are refused by `dataknobs_common.ontology.refuse_unbuildable_rungs`, the
same check both value-level doors run, so the three answers agree by being one
check rather than by being kept in step.

## Not yet here

Write-through: `RecordEntitySource` declares no write capability, which is a
property it states rather than a gap it hides.

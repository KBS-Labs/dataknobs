# Pack Composition

`dataknobs_common.packs` composes named, partial declarations into one
resolved declaration, in priority order, under rules each field declares
for itself.

A **pack** is a named, frozen, *partial* declaration. A **binding** selects
packs for one deployment and may tune them. **Resolution** folds the
selected packs — lowest priority first — field by field.

The module is deliberately vocabulary-free. It knows `name`, `priority`,
the `enabled` / `locked` binding flags, and a per-field merge rule; it
knows nothing about what the fields *mean*.
Domain packages define the vocabulary by subclassing `PackSpec`
(`dataknobs_bots.BehaviorPackSpec` is one such subclass).

## When to use this

Reach for packs when several independently-authored bundles of settings
must combine into one, and "which bundle wins" differs per field.

If you only need *one* source to win, you want something simpler:

| Need | Use |
|---|---|
| First source that answers supplies the whole answer | `dataknobs_common.resolver.CompositeResolver` |
| Merge attribute-by-attribute, first non-`None` per attribute | `dataknobs_llm.llm.model_profile.merge_partials` |
| **Every source contributes, per-field rules decide how** | **this module** |

Confusing the three is the easy mistake — they are all "merge" in casual
speech and none of them are the same operation.

## Quick start

```python
from dataclasses import dataclass, field
from types import MappingProxyType

from dataknobs_common.packs import MergeKind, PackRegistry, PackSpec


@dataclass(frozen=True)
class IngestPack(PackSpec):
    """How one corpus profile shapes an ingestion run."""

    chunker: str | None = None
    filters: tuple[str, ...] = ()
    limits: dict[str, int] = field(default_factory=dict)

    _COMPOSITION = MappingProxyType({
        "chunker": MergeKind.UNANIMOUS,
        "filters": MergeKind.CONCAT_UNIQUE,
        "limits": MergeKind.MERGE,
    })


registry = PackRegistry("ingest_profiles", IngestPack)
registry.register_pack(
    IngestPack(name="base", filters=("dedupe",), limits={"max_mb": 50})
)
registry.register_pack(
    IngestPack(name="regulated", priority=10,
               filters=("dedupe", "pii-redact"), limits={"max_mb": 10})
)

resolution = registry.resolve({"base": {}, "regulated": {"locked": True}})

resolution.packs           # ('base', 'regulated')
resolution.spec.filters    # ('dedupe', 'pii-redact')  — deduped concat
resolution.spec.limits     # {'max_mb': 10}  — highest priority wins the key
resolution.warnings        # (PackWarning(code='key_override', ...),)
```

## Declaring a spec

A subclass names its fields and declares how each one composes:

```python
@dataclass(frozen=True)
class MyPack(PackSpec):
    steps: tuple[str, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)

    _COMPOSITION = MappingProxyType({
        "steps": MergeKind.CONCAT,
        "options": MergeKind.MERGE,
    })
```

Four rules, all enforced:

1. The subclass **must** be `@dataclass(frozen=True)`. Immutability is what
   makes resolution safe to run repeatedly against a shared registry.
2. Every non-meta field **must** appear in `_COMPOSITION`. A field added
   later without a rule fails loudly instead of being silently dropped
   from composition.
3. Every non-meta field **must** have a default. A pack is a *partial*
   contribution — an unset field is one the pack does not speak to.
4. Every rule must be a `MergeKind` member or a callable.

Violations raise `ConfigurationError` when a `PackRegistry` is constructed
(or on the first `compose_packs` call), so an incoherent declaration
surfaces at wiring time rather than at first resolve.

`name` and `priority` are **meta fields** — they describe the pack rather
than contribute to composition, so they are excluded from `_COMPOSITION`
and from the fold.

If a subclass defines its own `__post_init__`, it **must** call
`super().__post_init__()` — see [Value normalization](#value-normalization).

## Merge rules

Every rule is applied as a left fold over the participating packs in
**ascending priority order**, so "last" means *highest priority* and
"first" means *lowest priority*.

| `MergeKind` | Behavior | Use when |
|---|---|---|
| `LAST_WINS` | Highest-priority value wins | A scalar knob a later pack should be able to retune |
| `FIRST_WINS` | Lowest-priority value wins | A value an early pack should be able to **pin** |
| `UNANIMOUS` | Every participating pack must agree, else raise | A claim two packs cannot both hold — disagreement is unsatisfiable, not resolvable |
| `CONCAT` | Sequences joined in fold order, duplicates kept | Order is behavior and a repeat is deliberate (middleware, hooks) |
| `CONCAT_UNIQUE` | Sequences joined, order-preserving dedup | Set-like names where a repeat is noise |
| `MERGE` | Shallow mapping merge, later packs win per key | Independent knobs contributed by several packs |

The composed shape does not depend on how many packs contributed: a
`CONCAT` / `CONCAT_UNIQUE` field is always a `tuple` and a `MERGE` field
always a `dict`, whether one pack participated or five.

### Participation: the rule that makes this work

A field **participates in the fold only when its contributed value differs
from that field's declared default.** "Unset" is "still at the default".

This is what makes `LAST_WINS` and `UNANIMOUS` behave: a pack that does not
mention a field must not clobber a pack that does. `CONCAT` and `MERGE` are
unaffected **as long as the field's declared default is empty**, since then
a non-participating contribution and an empty one amount to the same thing.

Give a `CONCAT` field a *non-empty* default and that stops holding: with

```python
steps: tuple[str, ...] = ("audit",)
```

a pack contributing exactly `("audit",)` is indistinguishable from one that
left the field alone, so it is skipped and the composed result has one
`"audit"` rather than two. Declare collection fields with an empty default
unless you specifically want that.

The rule exists because a spec is a frozen dataclass: every field is always
present, so "did not mention this" is not recoverable from the object and has
to be inferred from the value. The consequence is that **a pack cannot reset
a field to its default** — a pack declaring `chunker=None`, where `None` is
the default, contributes nothing.

Bindings are the exception, and deliberately so. A binding body is a partial
mapping you write by hand, so naming a field *is* the contribution:

```yaml
packs:
  ingest_fast:
    chunker: null      # explicitly clears it, even though null is the default
```

Participation is all that presence decides — the field's declared rule still
governs the outcome. A binding cannot take back a `FIRST_WINS` field its own
pack already pinned, and disagreeing with a `UNANIMOUS` field raises
`field_conflict` rather than clearing it.

Because whether that happens depends on the *pack's* content rather than on
anything visible in the binding, an override a rule discards is reported as
a `binding_override_ignored` warning rather than passing silently — the key
was typed by an operator and did nothing:

```python
@dataclass(frozen=True)
class PinPack(PackSpec):
    engine: str | None = None

    _COMPOSITION = MappingProxyType({"engine": MergeKind.FIRST_WINS})


pinned = PackRegistry("pinned", PinPack)
pinned.register_pack(PinPack(name="base", engine="v1"))

resolution = pinned.resolve({"base": {"engine": "v2"}})
resolution.spec.engine                        # 'v1' — the pack pinned it
[w.code for w in resolution.warnings]         # ['binding_override_ignored']
```

#### Making "explicitly the default" expressible in a pack

When a *pack* needs to contribute a value that happens to be the field's
default, declare `UNSET` as the default instead. Participation compares
against the declared default, so moving the "absent" marker off the domain's
own value space makes `None` an ordinary value:

```python
from dataknobs_common import UNSET

@dataclass(frozen=True)
class IngestPack(PackSpec):
    chunker: str | None = UNSET

    _COMPOSITION = MappingProxyType({"chunker": MergeKind.LAST_WINS})
```

```python
registry.register_pack(IngestPack(name="base", priority=0, chunker="fast"))
registry.register_pack(IngestPack(name="off", priority=10, chunker=None))
registry.register_pack(IngestPack(name="quiet", priority=10))

registry.resolve({"base": {}, "off": {}}).spec.chunker      # None  — explicit
registry.resolve({"base": {}, "quiet": {}}).spec.chunker    # 'fast' — silence
registry.resolve({"quiet": {}}).spec.chunker                # UNSET — untouched
```

The cost is that last line: a field no pack ever set reads back as `UNSET`
rather than a natural empty value, so consumers handle one extra case at the
point of use. `UNSET` is falsy, so `spec.chunker or fallback` still reads the
way it would with `None`.

Reach for this only on `LAST_WINS` / `FIRST_WINS` / `UNANIMOUS`, where a
default-valued contribution is meaningful. It buys nothing on `CONCAT` /
`CONCAT_UNIQUE` / `MERGE` declared with an empty default, where an empty
contribution is already a no-op.
Test it with `is UNSET` — identity is the contract, and it survives copy,
deepcopy, and pickle. It is an object rather than a string, so it does not
survive a JSON round-trip; a spec that must serialize to JSON should keep a
natural default and encode "explicitly off" as a domain value
(`chunker="none"`).

### Value shapes are checked

A value whose shape contradicts its rule raises `ConfigurationError` on
every contribution, including the first:

- `CONCAT` / `CONCAT_UNIQUE` require a **non-string sequence**. Without the
  check, `CONCAT` on a `str` would silently explode it into characters.
- `MERGE` requires a **mapping**, rather than raising an opaque `TypeError`
  deep in the fold.

### Custom rules — the callable hatch

When the built-in vocabulary does not cover a field, declare a plain
two-argument function instead. A `Reducer` is
`Callable[[Any, Any], Any]` — `(accumulated, next) -> value`:

```python
from dataknobs_config.inheritance import deep_merge

@dataclass(frozen=True)
class NestedPack(PackSpec):
    settings: dict[str, Any] = field(default_factory=dict)

    _COMPOSITION = MappingProxyType({
        "settings": deep_merge,   # recursive merge instead of shallow
    })
```

`deep_merge(base, override) -> dict` matches the `Reducer` signature
exactly, which is why recursive merge is available *through the hatch*
without `dataknobs-common` depending on `dataknobs-config`.

A custom reducer produces **no diagnostics** — `PackWarning`s are emitted
only by the built-in kinds, because the `Reducer` signature deliberately
carries neither the field name nor the contributing pack names.

It also gets no shape check, since the module cannot know what shapes it
accepts. If it raises, the exception is wrapped in a `ConfigurationError`
naming the field and the pack being folded in, so a mismatch does not
surface as a bare `TypeError` from inside the fold.

## Bindings

A binding body is **a partial spec plus two flags**:

```yaml
ingest_profiles:
  base: {}                        # enable with no overrides
  regulated:
    locked: true                  # a deployment must not turn this off
    limits: {max_mb: 5}           # tune one field
  experimental:
    enabled: false                # registered but not applied
```

| Key | Meaning |
|---|---|
| `enabled` | Select the pack. Default `true`. |
| `locked` | Assert the pack must not be turned off. Default `false`. |
| `priority` | Replace the pack's own priority (meta — replaced, not folded). |
| *any field name* | A field override, composed **spec first, binding second** under that field's declared rule. |

`locked: true` together with `enabled: false` is the contradiction this
mechanism exists to catch, and it raises.

Because field overrides go through the declared rule, a `MERGE` field
merges *over* the pack's values and a `CONCAT` field *appends* to them —
a binding adds to a pack, it does not replace it wholesale. One consequence
is deliberate: a `UNANIMOUS` field can be re-asserted by a binding but not
*changed*. A pack author's declared mechanism is not a deployment knob.

**Unknown binding keys are rejected, not ignored.** A typo'd `lockd: true`
that parsed as an unknown key would silently disable the safety contract.
Only composable fields plus `enabled` / `locked` / `priority` are bindable:
`name` is not (a pack is addressed by its own name), and neither is any
further descriptor a spec class adds to `_META_FIELDS`, since the fold has
no rule for it and would accept the key and then ignore it.

**Binding a pack requires it to be registered — including to disable it.**
Membership is checked before `enabled`, so `{"nope": {"enabled": false}}`
raises `unknown_pack` rather than resolving to nothing. Disabling is a
statement *about a known pack*, so a name that has been renamed or removed
should fail rather than quietly turn into a no-op.

An empty bindings mapping resolves to an all-default composed spec with no
applied packs. **Packs are opt-in.**

### Layering bindings

When more than one authority has a say — a platform baseline plus a
per-tenant overlay — merge the layers with `merge_bindings` rather than by
hand:

```python
from dataknobs_common.packs import merge_bindings

platform = {"regulated": {"locked": True}}
tenant = {"regulated": {"enabled": False}}

registry.resolve(merge_bindings(platform, tenant))
# PackResolutionError(reason="locked_pack_disabled")
```

Later layers win, per pack and per key. This is also what makes `locked`
more than a formality: inside a single body, `locked: true` next to
`enabled: false` is one author contradicting themselves; across layers it
is a platform baseline a tenant may not switch off.

Merging is **shallow within a body** — a later layer replaces the keys it
names. A later layer's `filters` replaces an earlier layer's rather than
appending, because layer precedence and per-field composition answer
different questions, and conflating them would make a binding's meaning
depend on which layer it came from. Compose *packs* to accumulate; layer
*bindings* to override.

### Loading from YAML

`PackSpec` extends
[`StructuredConfig`](structured-config.md), so specs load from config
mappings with `from_dict` and round-trip through `to_dict`:

```python
spec = IngestPack.from_dict({
    "name": "regulated",
    "priority": 10,
    "filters": ["dedupe", "pii-redact"],   # list -> tuple
    "limits": {"max_mb": 10},
})
```

## Ordering

Order is **ascending priority — the lower `priority` value folds first.**
This matches `dataknobs_common.callbacks.PriorityOrdering` ("lower fires
first") and the `priority=-100` guard idiom used elsewhere in the tree.

Because the fold runs low-to-high:

- `LAST_WINS` / `MERGE` give the **highest** priority the final say.
- `FIRST_WINS` / `UNANIMOUS` let the **lowest** (earliest) pack pin a value.
- `CONCAT` puts the **lowest**-priority pack's items first.

Ties break by **registration order** (FIFO). Registration order is
therefore part of the contract, not an implementation detail — and a tie
that *decides something* also emits a `priority_tie` warning, so it is
surfaced rather than left for a future registration-order change to alter
composition invisibly.

"Decides something" is the qualifier: the warning fires only when the tied
packs actually contend for a field. `priority` defaults to `0`, so binding
two packs without setting priorities is the ordinary case, and warning
there — where no field is contended and the order changes nothing — would
train consumers to filter the code out and cost them the case it exists
for.

## Diagnostics and failure

### Warnings — non-fatal

`PackResolution.warnings` carries structured `PackWarning`s rather than
bare strings, so a deployment can escalate a specific `code` to a hard
failure without pattern-matching prose:

| `code` | Emitted when |
|---|---|
| `priority_tie` | Two or more selected packs share a priority **and contend for a field** |
| `value_override` | `LAST_WINS` / `FIRST_WINS` discarded a differing value |
| `key_override` | `MERGE` overrode keys another pack had set |
| `binding_override_ignored` | A binding named a field whose declared rule then discarded the binding's value |

```python
for warning in resolution.warnings:
    if warning.code == "key_override":
        raise MyDeploymentError(str(warning))
    logger.warning("pack composition: %s", warning)
```

`PackWarning.packs` names the packs that **contributed** to the field's
current value, not every pack that mentioned it. Under `FIRST_WINS` a pack
whose value was discarded is left out, so it cannot show up as a source in
a later pack's warning.

A binding overriding *its own pack's* declared value produces no warning —
that is an intentional deployment act, not a diagnostic. The reverse does:
if the field's rule discards the binding's value, `binding_override_ignored`
reports it, because an operator typed a key that did nothing.

### Errors — two families, deliberately distinct

**Declaration/wiring errors** raise `ConfigurationError`: a field with no
declared rule, a rule for a field that does not exist, a non-meta field
with no default, a registered spec whose class adds fields the registry's
class cannot compose, a *spec's* value whose shape contradicts its rule, or
a custom reducer that fails. These are authoring bugs — the code is wrong,
not the deployment.

Most surface when the `PackRegistry` is constructed or the spec is
registered. Two cannot: a value's shape and a reducer's behavior are only
knowable once there is a value to check, so those surface at `resolve()`.

**Resolution errors** raise `PackResolutionError` (a `ConfigurationError`
subclass) carrying a machine-readable `reason`. These are operator input —
a binding a deployment supplied:

| `reason` | Cause |
|---|---|
| `unknown_pack` | A binding names a pack that is not registered |
| `unknown_binding_key` | A binding body carries a key that is neither a flag nor a composable field |
| `locked_pack_disabled` | `locked: true` with `enabled: false` |
| `field_conflict` | Two packs disagree on a `UNANIMOUS` field |
| `invalid_binding` | Malformed body, a non-boolean flag / non-integer priority, or a *binding's* value whose shape contradicts its rule |

The split follows the value's **origin**, not its symptom: the same bad
shape is an authoring bug inside a spec (written in code) and operator
input inside a binding (written in deployment config), so it raises a
different family in each case and only the latter carries a `reason`.

```python
from dataknobs_common.packs import PackResolutionError

try:
    resolution = registry.resolve(bindings)
except PackResolutionError as exc:
    if exc.reason == "unknown_pack":
        ...                      # probably a typo in deployment config
    raise
```

## Value normalization

`PackSpec.__post_init__` converts `list` values to `tuple`, `set` values to
`frozenset`, and copies `Mapping` values.

This is necessary because `StructuredConfig.from_dict` assigns non-config,
non-enum field values verbatim: a YAML list would otherwise land in a
`tuple[...]`-annotated field as a `list`, and a caller's dict would alias
into the "frozen" spec.

Normalization is **shallow**, and `frozen=True` blocks only *rebinding* a
field, never writing *through* one. So a `dict`-typed field value and every
container element are **read-only by convention**:

```python
spec = registry.get("base")
spec.limits["rps"] = 0        # legal Python — and it mutates the registry
limits = dict(spec.limits)    # do this instead
```

Sequence and set fields come out immutable by construction. Mappings do not,
and that asymmetry is deliberate: `MappingProxyType` cannot be deep-copied,
and `StructuredConfig.to_dict` delegates to `dataclasses.asdict`, which
deep-copies every field value — so freezing would trade a documented
convention for a broken serialization contract, and would booby-trap any
consumer calling `dataclasses.asdict` on a spec directly.

## Composing specs you already hold

`PackRegistry.resolve` is the priority-ordered, binding-aware path.
`compose_packs` is the lower-level primitive it delegates to, for callers
that already hold specs in the order they want:

```python
from dataknobs_common.packs import compose_packs

composed, warnings = compose_packs(
    [base_spec, regulated_spec],   # caller supplies the order; no sorting
    IngestPack,
)
```

## Purity and threading

`PackRegistry.resolve` is **pure and synchronous**: it mutates neither the
registry, the registered specs, nor the `bindings` mapping, and resolving
the same bindings twice yields equal results. Nothing in the module does
I/O.

`PackRegistry` extends `dataknobs_common.registry.Registry`, inheriting
thread-safety, optional metrics, and structural conformance to
`BackendRegistry`. It extends `Registry` rather than `PluginRegistry`
because packs are eagerly registered *declarations*, not
lazily-constructed backends — a distinction the
[Plugin Registry guide](plugin-registry.md) draws in its opening.

A registry holds **exactly one spec class**, and composition is defined
entirely by that class's `_COMPOSITION`. Registering a subclass that adds
its own fields is rejected: the composed result is an instance of the
registry's class, so a field only the subclass declares has no rule and
nowhere to land, and would be dropped silently. Subclassing to add
*behavior* is fine — it is extra fields that cannot be honoured.

**No module-level singleton is provided, and consumers should not create
one.** A pack binding is a per-deployment decision; a process-global pack
registry is a multi-tenant hazard. Construct one and own it.

## API summary

| Name | Role |
|---|---|
| `PackSpec` | Frozen base a consumer subclasses to name its fields |
| `MergeKind` | The six built-in per-field composition rules |
| `UNSET` | Sentinel default letting a pack contribute a default-valued field |
| `Reducer` | `(acc, next) -> value` — the custom-rule escape hatch |
| `CompositionRule` | `MergeKind \| Reducer` — what `_COMPOSITION` values may be |
| `PackRegistry` | Registry of specs keyed by `spec.name`, plus `resolve()` |
| `PackResolution` | `applied`, `spec`, `warnings`, and the `packs` name tuple |
| `PackWarning` | Structured non-fatal diagnostic (`code`, `message`, `packs`, `field`) |
| `PackResolutionError` | Fail-closed rejection carrying `reason` |
| `compose_packs` | Fold an already-ordered sequence of specs |
| `merge_bindings` | Merge ordered binding layers, later layers winning |

All are exported from the top-level `dataknobs_common` namespace.

## See also

- [Structured Config](structured-config.md) — the `from_dict` / `to_dict`
  base `PackSpec` builds on
- [Plugin Registry](plugin-registry.md) — the registry family
  `PackRegistry` belongs to
- **Behavior Packs** (`dataknobs-bots`) — the bot-flavored vocabulary
  built on this module

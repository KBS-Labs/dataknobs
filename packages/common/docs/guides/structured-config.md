# Structured Configuration

`dataknobs_common.structured_config` provides two primitives that
together replace the kwarg-splat pattern of `ConfigurableBase`:

1. **`StructuredConfig`** — a frozen-dataclass base with auto-derived
   `from_dict()` / `to_dict()` via `dataclasses.fields()` introspection,
   including recursion into nested/collection `StructuredConfig` fields.
2. **`StructuredConfigConsumer[ConfigT]`** — a generic mixin providing
   typed/dict/loose-kwarg dispatch in a single `__init__`, a typed
   `self.config` property, sync (`from_config`) and async
   (`from_config_async`) entry points, `_setup()` / `_ainit()` hooks,
   and cooperative `super().__init__()` so it composes into a
   multiple-inheritance hierarchy.

These primitives generalize the per-backend hand-rolled pattern in
`dataknobs_common.events` so every future "object configured by a
typed dataclass" pair can share one implementation. Drift between the
dataclass field set and the consumer's construction surface becomes
structurally impossible — the field set IS the construction surface.

## When to use it

Reach for `StructuredConfig` + `StructuredConfigConsumer` when:

- A class is configured by a structured set of typed knobs.
- Construction needs to accept BOTH typed configs and loose dicts
  (registry-driven construction, YAML-loaded configs).
- The class is part of a registry where a factory should not have to
  enumerate kwargs.

Reach for plain `Serializable` (no nominal inheritance) when the
object is **data interchange** rather than configuration —
provenance, audit logs, request/response records.

## Quick example

```python
from dataclasses import dataclass
from typing import ClassVar
from dataknobs_common.structured_config import (
    StructuredConfig,
    StructuredConfigConsumer,
)


@dataclass(frozen=True)
class WidgetConfig(StructuredConfig):
    name: str = "default"
    size: int = 1


class Widget(StructuredConfigConsumer[WidgetConfig]):
    CONFIG_CLS: ClassVar[type[WidgetConfig]] = WidgetConfig

    def _setup(self) -> None:
        self.area = self._config.size ** 2


# All four shapes reach the same internal state:
Widget()
Widget(WidgetConfig(name="x", size=4))
Widget({"name": "x", "size": 4})
Widget(name="x", size=4)

# `from_config` is the recommended programmatic entry point:
Widget.from_config({"name": "x", "size": 4})
```

## `StructuredConfig` API

### `from_dict(config)` (classmethod)

Auto-derived projection. Each `@dataclass` field is read from `config`
by name; absent fields use their declared default (or
`default_factory`). Unknown keys in the dict are ignored — useful for
registry-routing keys like `"backend"`. The caller's dict is
shallow-copied before normalization, so it isn't mutated.

A field whose declared type is (or contains) a `StructuredConfig`
subclass is rebuilt recursively from its raw dict shape — see
[Nested-config composition](#nested-config-composition). All other
fields are assigned verbatim.

### `_normalize_dict(cls, raw)` (classmethod, override hook)

Default identity. Override when the input dict's shape differs from
the field set — for example, when a config dict's `connection_string`
key must be derived from a `DATABASE_URL` env-var or assembled from
individual `host`/`port`/`database` keys. The argument is the
shallow-copied dict; the override may mutate it freely.

```python
@dataclass(frozen=True)
class RenamedConfig(StructuredConfig):
    new_field: str = "default"

    @classmethod
    def _normalize_dict(cls, raw):
        if "legacy_key" in raw and "new_field" not in raw:
            raw["new_field"] = raw.pop("legacy_key")
        return raw
```

### `to_dict()`

Symmetric serialization via `dataclasses.asdict`. Round-trip property —
`type(cfg).from_dict(cfg.to_dict()) == cfg` — holds for flat configs
and for nested configs alike: `asdict` recurses into nested dataclasses
and `from_dict` recurses back into the matching field types, so the two
are symmetric for every statically-typed nesting shape (see
[Nested-config composition](#nested-config-composition)). No
`_normalize_dict` override is required for nesting.

### `__post_init__` validation

Per-class invariants belong in `__post_init__` (not `from_dict`):

```python
@dataclass(frozen=True)
class StrictConfig(StructuredConfig):
    name: str = ""

    def __post_init__(self):
        if not self.name:
            raise ValueError("name must be non-empty")
```

`from_dict({})` will trigger the validator just like direct
construction does.

## `StructuredConfigConsumer[ConfigT]` API

### `CONFIG_CLS` (ClassVar)

Required: declare the concrete `StructuredConfig` subclass.

```python
class Widget(StructuredConfigConsumer[WidgetConfig]):
    CONFIG_CLS: ClassVar[type[WidgetConfig]] = WidgetConfig
```

### `__init__(config, **kwargs)` dispatch

| Call                                  | Behavior                                              |
| ------------------------------------- | ----------------------------------------------------- |
| `Widget()`                            | `WidgetConfig()` (all defaults)                       |
| `Widget(WidgetConfig(...))`           | typed config used directly                            |
| `Widget({"name": "x"})`               | dict passed to `from_dict`                            |
| `Widget(name="x")`                    | kwargs merged into dict, passed to `from_dict`        |
| `Widget(None, name="x")`              | same as kwargs-only — `None` config is a no-op        |
| `Widget(WidgetConfig(...), name="x")` | `TypeError` — ambiguous, refuse to guess              |
| `Widget(42)`                          | `TypeError` — non-Mapping non-`ConfigT` is invalid    |

### `from_config(config)` (classmethod)

Recommended programmatic-construction entry point. Registry factories
should be one-line wrappers:

```python
def _create_widget(config: dict) -> Widget:
    return Widget.from_config(config)
```

### `from_config_async(config)` (classmethod)

Async construction entry point: the same sync assembly as `from_config`
(typed/dict dispatch through `_coerce_config`), followed by
`await obj._ainit()`. Use it for consumers whose initialization is
asynchronous — databases that connect eagerly, LLM-backed bots,
knowledge-base warmup. Synchronous consumers ignore it and use
`from_config`.

```python
widget = await Widget.from_config_async({"name": "x", "size": 4})
```

### `self.config` (property)

Read-only typed view. Backed by the frozen `ConfigT` so runtime
mutation is rejected.

### `_setup()` (sync override hook)

Default no-op. Override to initialize derived attributes computed
from `self.config.*` — connection placeholders, lock/handle
initialization, etc. Field normalization belongs in the config
dataclass (`_normalize_dict` / `__post_init__`), not here. Called once
during `__init__` (both the sync and async construction paths) after
`self._config` is established.

### `_ainit()` (async override hook)

Default no-op. Override for awaitable setup that cannot run in the
synchronous `_setup` — connection establishment, provider warmup, async
collaborator construction. Runs exactly once, after `_setup`, **only**
on the `from_config_async` path; the synchronous `__init__` /
`from_config` path does not run it.

### Cooperative multiple inheritance

`__init__` calls `super().__init__()` after building `self._config` and
before `_setup()`. For a single-base consumer `super()` is `object` and
this is a no-op. When mixing the consumer in alongside other bases, list
`StructuredConfigConsumer` **first** so its `__init__` is the entry
point, and ensure the remaining bases accept `__init__()` with no
required args (they should expose `_setup`, not a competing config
ctor). `assert_structured_config_consumer` pins this ordering.

## Patterns

### Back-compat positional shortcuts

The mixin's `__init__` is `(self, config, **kwargs)`. If a legacy
public API exposed a positional shortcut you cannot drop, override
`__init__` thin and forward to `super().__init__(config=...)`:

```python
class LegacyBus(StructuredConfigConsumer[LegacyConfig]):
    CONFIG_CLS: ClassVar[type[LegacyConfig]] = LegacyConfig

    def __init__(
        self,
        dsn: str | None = None,
        *,
        config: LegacyConfig | Mapping[str, Any] | None = None,
    ) -> None:
        if dsn is not None:
            if isinstance(config, LegacyConfig):
                raise TypeError("cannot mix typed config with `dsn`")
            merged = dict(config or {})
            merged["dsn"] = dsn
            config = merged
        super().__init__(config=config)
```

If you do this, also override `from_config` so the classmethod
delivers the typed config via the `config=` slot rather than the
positional one:

```python
@classmethod
def from_config(cls, config):
    if isinstance(config, cls.CONFIG_CLS):
        return cls(config=config)
    return cls(config=cls.CONFIG_CLS.from_dict(config))
```

### Nested-config composition

`from_dict` recurses automatically into fields whose declared type is
(or contains) a `StructuredConfig` subclass — no `_normalize_dict`
override is needed:

```python
@dataclass(frozen=True)
class OuterConfig(StructuredConfig):
    inner: InnerConfig = dataclasses.field(default_factory=InnerConfig)
    optional: InnerConfig | None = None
    many: list[InnerConfig] = dataclasses.field(default_factory=list)
    by_name: dict[str, InnerConfig] = dataclasses.field(default_factory=dict)
    grouped: dict[str, list[InnerConfig]] = dataclasses.field(
        default_factory=dict
    )


cfg = OuterConfig.from_dict({
    "inner": {"value": 1},
    "optional": None,
    "many": [{"value": 2}, {"value": 3}],
    "by_name": {"a": {"value": 4}},
    "grouped": {"g": [{"value": 5}]},
})
# cfg.inner is an InnerConfig; cfg.many is list[InnerConfig]; etc.
```

The supported shapes are `SubCfg`, `SubCfg | None`, `list[SubCfg]`,
`tuple[SubCfg, ...]`, `set[SubCfg]`, `dict[K, SubCfg]`, and
`dict[K, list[SubCfg]]`. Recursion is bounded by the static field-type
graph (not by runtime data), and a value that is already a typed
instance passes through untouched. Use `_normalize_dict` only for
genuine shape massaging (key renames, connection-string assembly) — not
for nesting.

**Polymorphic selection is out of scope here.** When a section's
concrete type is chosen by a discriminator key (a `memory` block that is
a *buffer* vs *vector* vs *summary* strategy), type the field as the
abstract sub-config and dispatch the discriminator in the consuming
subsystem's factory / registry. Keeping a subclass registry out of
`from_dict` avoids dragging a registry into this zero-dependency
primitive and duplicating the `class`/`factory`/`type` dispatch that the
`dataknobs-config` object-graph layer already owns.

### Environment-variable substitution

`StructuredConfig` lives in `dataknobs-common`, the lowest workspace
layer; `substitute_env_vars` lives in the higher-layer
`dataknobs-config`. To avoid an inverted dependency, consumers call
`substitute_env_vars` themselves before `from_dict`:

```python
from dataknobs_config import substitute_env_vars

raw = {"name": "${WIDGET_NAME:default}", "size": "${WIDGET_SIZE:1}"}
cfg = WidgetConfig.from_dict(substitute_env_vars(raw, type_coerce=True))
```

## Testing helpers

### `assert_structured_config_consumer(consumer_cls)`

Unified parity guard combining structural checks:

1. `consumer_cls` declares `CONFIG_CLS`.
2. `CONFIG_CLS` is a `StructuredConfig` subclass.
3. The dataclass field set matches the consumer's ctor surface.
4. The mixin precedes other bases in the MRO (so its `__init__` is the
   construction entry point) — unless the consumer defines its own
   `__init__` (the documented back-compat shortcut).
5. An overridden `from_config_async` routes through `_coerce_config`.
6. (Optional) A registry factory passed via `expected_factory=`
   delegates to `from_config`.

```python
from dataknobs_common.testing import assert_structured_config_consumer

def test_widget_uses_structured_config_consumer():
    assert_structured_config_consumer(Widget)
```

### `assert_structured_config_roundtrip(cfg)`

Property assertion: `type(cfg).from_dict(cfg.to_dict()) == cfg`. Holds
for flat configs and for nested configs alike (see [Nested-config
composition](#nested-config-composition)) — `to_dict`/`from_dict`
recurse symmetrically, so no `_normalize_dict` override is required for
nesting.

```python
from dataknobs_common.testing import assert_structured_config_roundtrip

def test_widget_config_roundtrips():
    assert_structured_config_roundtrip(WidgetConfig(name="x", size=4))
```

## Construction in the object-graph layer

`StructuredConfig` + `StructuredConfigConsumer` own how a **single**
object is built from its own config. Wiring configured objects into a
**graph** — references, environment-specific bindings, and polymorphic
`class`/`factory`/`type` selection — is owned by the higher-layer
`dataknobs-config` (`Config.build_object`, `ObjectBuilder`,
`ConfigBindingResolver`). That layer calls the per-object contract:
`build_object` prefers a target's `from_config`, and the async
counterparts prefer `from_config_async`:

```python
# sync graph construction
obj = config.build_object("xref:widget[primary]")

# async graph construction — awaits from_config_async when defined,
# else a factory's create_async, else falls back to the sync path.
obj = await config.build_object_async("xref:widget[primary]")
```

`ConfigBindingResolver.resolve_async` follows the same preference. The
dependency direction is one-way (`dataknobs-config` depends on
`dataknobs-common`), so the primitive never imports the graph layer; the
graph layer calls the primitive's entry points.

## Relationship to other abstractions

| Abstraction                | Role                                       | When                                              |
| -------------------------- | ------------------------------------------ | ------------------------------------------------- |
| `StructuredConfig`         | typed config dataclass                     | configuration objects                             |
| `StructuredConfigConsumer` | mixin for classes configured by a config   | classes built from a `StructuredConfig`           |
| `Serializable`             | structural protocol for `to_dict`/`from_dict` | data interchange (records, events, audit logs) |
| `ConfigurableBase`         | kwarg-splat predecessor (soft-deprecated)  | existing consumers; new code uses the mixin       |

`StructuredConfig` instances structurally satisfy `Serializable` — the
two are complementary, and you do not need to nominally inherit from
`Serializable` to be treated as one.

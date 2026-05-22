# Structured Configuration

`dataknobs_common.structured_config` provides two primitives that
together replace the kwarg-splat pattern of `ConfigurableBase`:

1. **`StructuredConfig`** — a frozen-dataclass base with auto-derived
   `from_dict()` / `to_dict()` via `dataclasses.fields()` introspection.
2. **`StructuredConfigConsumer[ConfigT]`** — a generic mixin providing
   typed/dict/loose-kwarg dispatch in a single `__init__`, a typed
   `self.config` property, a one-line `from_config()` classmethod, and
   a `_setup()` hook for subclass initialization.

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
`type(cfg).from_dict(cfg.to_dict()) == cfg` — holds for flat configs,
and for nested configs whose `_normalize_dict` rebuilds each nested
`StructuredConfig` field from its dict (see [Nested-config
composition](#nested-config-composition)). It does **not** hold for a
nested field left as a raw dict: `asdict` recurses into nested
dataclasses but `from_dict` does not, so without the `_normalize_dict`
override the recovered field is a `dict`, not the typed sub-config.

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

### `self.config` (property)

Read-only typed view. Backed by the frozen `ConfigT` so runtime
mutation is rejected.

### `_setup()` (override hook)

Default no-op. Override to initialize derived attributes computed
from `self.config.*` — connection placeholders, lock/handle
initialization, etc. Field normalization belongs in the config
dataclass (`_normalize_dict` / `__post_init__`), not here. Called once
during `__init__` after `self._config` is established.

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

`from_dict` does NOT automatically recurse into fields whose declared
type is a `StructuredConfig` subclass. For nested configs, override
`_normalize_dict` to call the sub-config's `from_dict`:

```python
@dataclass(frozen=True)
class OuterConfig(StructuredConfig):
    inner: InnerConfig = dataclasses.field(default_factory=InnerConfig)

    @classmethod
    def _normalize_dict(cls, raw):
        if isinstance(raw.get("inner"), Mapping):
            raw["inner"] = InnerConfig.from_dict(raw["inner"])
        return raw
```

This deliberate scoping keeps type introspection bounded — generic
auto-projection of `T | None`, `list[T]`, `dict[K, T]` would require
careful handling of forward references and was kept out of the base
to avoid drift modes.

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

Unified parity guard combining four structural checks:

1. `consumer_cls` declares `CONFIG_CLS`.
2. `CONFIG_CLS` is a `StructuredConfig` subclass.
3. The dataclass field set matches the consumer's ctor surface.
4. (Optional) A registry factory passed via `expected_factory=`
   delegates to `from_config`.

```python
from dataknobs_common.testing import assert_structured_config_consumer

def test_widget_uses_structured_config_consumer():
    assert_structured_config_consumer(Widget)
```

### `assert_structured_config_roundtrip(cfg)`

Property assertion: `type(cfg).from_dict(cfg.to_dict()) == cfg`. Holds
for flat configs and for nested configs whose `_normalize_dict` rebuilds
each nested `StructuredConfig` field (see [Nested-config
composition](#nested-config-composition)); a nested field left as a raw
dict will fail the assertion.

```python
from dataknobs_common.testing import assert_structured_config_roundtrip

def test_widget_config_roundtrips():
    assert_structured_config_roundtrip(WidgetConfig(name="x", size=4))
```

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

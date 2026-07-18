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
[Nested-config composition](#nested-config-composition). A field whose
declared type is (or contains) an `Enum` subclass is rebuilt from its
member value, so `{"mode": "fast"}` loaded from YAML/JSON becomes
`Mode.FAST` rather than the bare string `"fast"` (this works through the
same container and `| None` shapes as nested-config recursion; an
unrecognised value passes through unchanged so the constructor /
`__post_init__` owns the diagnostic). `StrEnum` / `IntEnum` members pass
through unchanged. All other fields are assigned verbatim.

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

This is the *in-process* form: `Enum` fields render as their members
(not their `.value`), and live callables / `type` objects round-trip by
identity — so the output is not necessarily JSON-serialisable.

### `to_json_dict()`

Like `to_dict()`, but every `Enum` member (at any depth, including inside
list/tuple/dict containers and nested configs) is rendered as its
`.value`. Because `from_dict` coerces those raw values back to members,
the round-trip survives JSON:

```python
import json
restored = type(cfg).from_dict(json.loads(json.dumps(cfg.to_json_dict())))
assert restored == cfg   # for configs whose other fields are JSON-native
```

Enum normalisation is the only transformation it adds: configs carrying
live callables or `type` objects still hold those verbatim and remain no
more JSON-serialisable than via `to_dict()`, and `set` / `frozenset`
fields are left untouched (JSON has no set type). The enum normalisation
is performed by the shared `dataknobs_common.serialization.jsonify`
utility (re-exported as `dataknobs_common.jsonify`), reusable on any
JSON-shaped structure.

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

### Secret redaction (`_SENSITIVE_FIELDS`)

A config that carries a credential — an API key, or a connection string
that embeds a password — masks it in `repr` by listing the field
name(s) in a `_SENSITIVE_FIELDS` class variable:

```python
@dataclass(frozen=True)
class ServiceConfig(StructuredConfig):
    host: str = "localhost"
    api_key: str | None = None

    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset({"api_key"})

repr(ServiceConfig(host="db", api_key="sk-live-123"))
# "ServiceConfig(host='db', api_key='***')"  ← masked
```

That declaration is the **only** thing required — redaction is
automatic. `StructuredConfig.__init_subclass__` installs a redacting
`__repr__` on every subclass *before* its `@dataclass` decorator runs,
so:

- a dataclass-*generated* (plaintext) `__repr__` is never produced, and
- the redacting repr is **inheritance-safe** — it is reinstalled at
  every level, so an intermediate `@dataclass` base in the MRO cannot
  shadow it. Declaring `_SENSITIVE_FIELDS` on a shared base
  (e.g. `S3DatabaseConfigBase`) covers every leaf that inherits it.

Semantics:

- A non-empty value of a listed field renders as `'***'`.
- `None` and `""` render verbatim — an unset credential is not a secret,
  and masking `""` would falsely imply one is configured.
- `to_dict()` is **never** redacted (round-trip must reconstruct the
  real value); redaction is display-only, keeping secrets out of logs
  that interpolate `repr(config)`, tracebacks, and pytest failure output.
- A config with no `_SENSITIVE_FIELDS` renders byte-for-byte identically
  to the standard dataclass repr — the mechanism is inert unless a
  subclass opts in.

#### Nested-mapping redaction (raw polymorphic sections)

Redaction also descends into raw `Mapping`/`list` field values. A
polymorphic/discriminated section is held as a raw `dict` at the parent
layer because its schema is owned by another package and dispatched by a
discriminator key (e.g. `vector_store={"backend": "pgvector",
"connection_string": "postgresql://u:pw@h/db"}`,
`embedding={"provider": "openai", "api_key": "sk-…"}`, an `llm` provider
dict). The parent has no typed field for the credential, so it is masked
by **interior key name** instead:

```python
@dataclass(frozen=True)
class RAGConfig(StructuredConfig):
    vector_store: dict[str, Any] = field(default_factory=dict)
    # no _SENSITIVE_FIELDS entry needed for connection_string

repr(RAGConfig(vector_store={"backend": "pgvector",
                             "connection_string": "postgresql://u:pw@h/db"}))
# "RAGConfig(vector_store={'backend': 'pgvector',
#  'connection_string': '***'})"  ← masked, no per-class config
```

The interior key set is a module default —
`{api_key, password, connection_string, client_secret, secret_key,
secret_access_key, access_token, refresh_token, auth_token, bearer_token,
aws_access_key_id, aws_secret_access_key, aws_session_token}` — unioned
with the class's `_SENSITIVE_FIELDS`, so the known leaks close with zero
per-class configuration. Interior matching is **exact and
case-insensitive** (a benign `access_token_expiry` is *not* masked,
despite containing `access_token`), **truthy-only** (an empty/absent
nested credential renders verbatim), and **depth-bounded** (default 6,
ample for the ~2-3-level real shapes). Like the scalar path it is
**display-only** — `to_dict()` and round-trip are unchanged. This is the
redaction-side complement to the subsystem registries: both act on a
key-name handle because the section schema is opaque at the parent layer.

**Why only unambiguous compound names, not bare `token` / `secret`.**
Interior matching targets raw dict sections whose schema is owned by
*another* package. A false positive there masks a benign key the consumer
**cannot rename** (it is a third-party schema) and **cannot remove** from
the default set (it is frozen; `_SENSITIVE_FIELDS` only *adds*). So the
default set deliberately excludes the bare generics `token` (NLP tokens,
pagination/page tokens, CSRF tokens) and `secret` (a `secret` flag, a
`secret` name reference to a vault) in favour of the compound `*_token` /
`*_secret*` names above, which are almost never benign.

Extending the set:

- **Per class** — a config whose own opaque section uses a non-default
  credential key adds it to that class's `_SENSITIVE_FIELDS` (it doubles as
  the scalar field-name set and the interior-key extension).
- **Process-globally** — a consumer with custom opaque sections that share
  a credential key calls `register_sensitive_interior_key("vault_ref")`
  once at startup; it masks that key inside every `StructuredConfig`
  subclass thereafter. Registration is **add-only** — there is no removal,
  because redaction is a fail-closed security feature and configuration
  must never be able to switch credential masking *off*.

```python
from dataknobs_common import register_sensitive_interior_key

register_sensitive_interior_key("vault_ref")  # masked everywhere, add-only
```

Raising the depth bound: a subclass whose raw section nests deeper than the
default 6 levels may raise its own bound via the `_MAX_REDACT_DEPTH` ClassVar.
Like the key levers, this is **raise-only** — setting it *below* the floor
shrinks the masked region (a configuration-time reduction of protection) and
is rejected with `ValueError` at class definition:

```python
@dataclass(frozen=True)
class DeepConfig(StructuredConfig):
    section: dict[str, Any] = field(default_factory=dict)
    _MAX_REDACT_DEPTH: ClassVar[int] = 8   # OK: deeper masking
    # _MAX_REDACT_DEPTH: ClassVar[int] = 2 # ValueError: below the floor of 6
```

Schema-*precise* redaction for these sections (typing them so the
constructed child self-redacts) is the future typed-nesting work; until
then the interior-key descent is the standing mechanism, and would remain
as defense-in-depth.

### Polymorphic-section validation (`validate()` + `config_registries`)

`from_dict` is deliberately tolerant: it ignores keys matching no field
(registry-routing keys like `backend` / `provider` pass through) and does
**not** look inside a raw-dict section whose schema is owned by another
package (`vector_store`, `embedding`, `llm`). `validate()` is the opt-in
companion that checks those polymorphic sections **without constructing the
runtime objects** — the use case is the gap between *parse* and
*construction*: CI config-linting, write-time validation in a multi-tenant
config store, config editors. A flow that builds the object right after
parsing already gets a fast failure from its construction factory; this is
for flows that parse but do not (yet) build.

A config opts in by declaring `_polymorphic_fields` (field name → binding
name) and relying on the field's owning package to register a resolver
under that binding name. The binding is a **string** plus a runtime
registration, so adopting validation adds **no** import of the child config
type — it stays coupling-free:

```python
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from dataknobs_common.structured_config import StructuredConfig

@dataclass(frozen=True)
class RAGConfig(StructuredConfig):
    # "vector_store" field is governed by the "vector_store" binding,
    # whose resolver dataknobs-data registers on import. No data import here.
    _polymorphic_fields: ClassVar[Mapping[str, str]] = {
        "vector_store": "vector_store"
    }
    vector_store: dict[str, Any] = field(default_factory=dict)
```

`validate()` resolves each declared section's concrete config class via
`config_registries`, calls its `from_dict` purely to surface field-level
errors (discarding the result — the field stays a raw dict), then validates
the built child so **one `parent.validate()` validates the whole tree**
(a bot config's knowledge-base section down to its vector-store section).
Statically-typed nested `StructuredConfig` fields are recursed into too.

Behavior:

| Situation | Behavior |
|---|---|
| no `_polymorphic_fields` (non-adopter) | no-op |
| section value empty (`{}` / `None`) | skipped (the default / `from_components` path) |
| binding has no registered resolver | skipped + `logger.debug` (fail-soft: cause is import order; the test guard below catches real drift) |
| resolver returns `None` (unknown discriminator) | raises `ConfigurationError` — the headline win: a typo'd `backend` caught at config-lint time |
| resolver returns `SKIP_VALIDATION` (recognized, but no typed config — e.g. a bare-callable backend) | skipped + `logger.debug` *in `validate()`* — rejecting a valid, constructible backend would be a false positive. (A resolver may itself `logger.warning` before returning the sentinel, so operators see an actionable "give this backend a `CONFIG_CLS`" signal; the two log sites and levels are intentional.) |
| resolver returns a config class | dry-run `from_dict` (surfaces child field errors) + recurse |

The parse-without-construct CI-lint usage:

```python
cfg = RAGConfig.from_dict(loaded_yaml)   # cheap parse — does not build the KB
cfg.validate()                           # raises on a bad/unknown vector_store
```

`validate()` is **never** auto-called by `from_dict` or by construction —
`from_dict`'s tolerant contract is preserved, and construction already fails
fast.

#### Registering a resolver (`config_registries`)

`config_registries` is a process-global `Registry[ConfigClassResolver]`
that `dataknobs-common` owns but (having `dependencies = []`) cannot
populate — the **registry-of-registries** seam. Each package that owns a
polymorphic section registers its resolver eagerly at import. A resolver
maps the section's raw dict to its concrete config class (`None` for an
unknown discriminator, or `SKIP_VALIDATION` when the discriminator is
recognized but has no typed config to check) and **must delegate to the
section's own construction registry** so validation and construction cannot
drift:

```python
# dataknobs_data/vector/stores/__init__.py
import logging

from dataknobs_common.structured_config import (
    SKIP_VALIDATION, StructuredConfig, config_registries,
)

logger = logging.getLogger(__name__)

def _resolve_vector_store_config_cls(raw):
    backend = raw.get("backend", "memory")
    store_cls = vector_backends.get_factory(backend)   # the construction registry
    if store_cls is None:
        return None                                     # unknown backend -> raise
    config_cls = getattr(store_cls, "CONFIG_CLS", None) # no independent table
    if isinstance(config_cls, type) and issubclass(config_cls, StructuredConfig):
        return config_cls
    # Registered (e.g. bare callable) but untyped -> skip. Warn first so an
    # operator sees an actionable signal rather than a silent skip: this
    # backend exists but isn't validatable until it grows a CONFIG_CLS.
    logger.warning(
        "Backend %r is registered but exposes no CONFIG_CLS; "
        "validate() will skip its section. Give it a CONFIG_CLS "
        "to make the section validatable.",
        backend,
    )
    return SKIP_VALIDATION

config_registries.register(
    "vector_store", _resolve_vector_store_config_cls, allow_overwrite=True
)
```

`SKIP_VALIDATION` matters for construction registries that accept
bare-callable backends (no `CONFIG_CLS` to read): such a backend is valid
and constructible, so returning `None` (→ raise) would be a false positive.
A closed registry of config-bearing classes never needs it. A production
resolver should `logger.warning` before returning the sentinel (as above), so
operators know a registered backend is silently unvalidated and can act —
`validate()` itself only emits a `logger.debug` for the skip.

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

### `from_config(config, **components)` (classmethod)

Recommended programmatic-construction entry point. Registry factories
should be one-line wrappers:

```python
def _create_widget(config: dict) -> Widget:
    return Widget.from_config(config)
```

Optional keyword `components` carry injected collaborators (see
[Collaborator injection](#collaborator-injection)); the collaborator-free
call `from_config(config)` is unchanged.

### `from_config_async(config, **components)` (classmethod)

Async construction entry point: the same sync assembly as `from_config`
(typed/dict dispatch through `_coerce_config`), followed by
`await obj._ainit(**components)`. Use it for consumers whose
initialization is asynchronous — databases that connect eagerly,
LLM-backed bots, knowledge-base warmup. Synchronous consumers ignore it
and use `from_config`. Injected collaborators passed as keyword
`components` are delivered to `_ainit` (see
[Collaborator injection](#collaborator-injection)).

```python
widget = await Widget.from_config_async({"name": "x", "size": 4})
```

#### Async-canonical construction (the `from_config` async delegator)

`from_config_async` is the canonical entry point for an asynchronously
constructed object — it is the only path that runs `_ainit`. The base
`from_config` stays synchronous and never runs `_ainit`. An object whose
canonical construction is async **and** that must keep a public
`await X.from_config(...)` API may override `from_config` with a one-line
async delegator — the blessed counterpart to the
[back-compat `__init__` shortcut](#back-compat-positional-shortcuts):

```python
@classmethod
async def from_config(cls, config, **components) -> Self:
    return await cls.from_config_async(config, **components)
```

This is not a divergence: it routes straight through
`from_config_async`, so `_coerce_config`, `__init__(config)`, `_setup`,
and `_ainit` all run. A consumer with explicit injection kwargs keeps
them and forwards them as components:

```python
@classmethod
async def from_config(cls, config, *, llm=None, middleware=None) -> Bot:
    return await cls.from_config_async(config, llm=llm, middleware=middleware)
```

Footgun: an async `from_config` override *removes* the synchronous
half-built path for that class — `X.from_config(...)` now returns a
coroutine — which is exactly what an async-built object wants. A caller
that forgets the `await` is not left without a signal: a static type
checker flags any use of the result as an instance (its inferred type is
`Coroutine[..., Self]`, not `Self`), and the interpreter emits
`RuntimeWarning: coroutine '...from_config' was never awaited` when the
orphaned coroutine is garbage-collected. The parity guard requires an
async `from_config` override to delegate to `from_config_async`, and a
sync override to route through `_coerce_config` (see
[`assert_structured_config_consumer`](#testing-helpers)).

### `self.config` (property)

Read-only typed view. Backed by the frozen `ConfigT` so runtime
mutation is rejected.

### `self.components` (property)

Read-only mapping of injected collaborators (empty for config-only
construction). Populated when collaborators are passed through
`from_config` / `from_config_async` / `from_components` — see
[Collaborator injection](#collaborator-injection).

### `_setup()` (sync override hook)

Default no-op. Override to initialize derived attributes computed
from `self.config.*` — connection placeholders, lock/handle
initialization, etc. Field normalization belongs in the config
dataclass (`_normalize_dict` / `__post_init__`), not here. Called once
during `__init__` (both the sync and async construction paths) after
`self._config` is established.

### `_ainit(**components)` (async override hook)

Default no-op. Override for awaitable setup that cannot run in the
synchronous `_setup` — connection establishment, provider warmup, async
collaborator construction. Runs exactly once, after `_setup`, **only**
on the `from_config_async` path; the synchronous `__init__` /
`from_config` path does not run it. Injected collaborators (see
[Collaborator injection](#collaborator-injection)) are delivered here as
keyword arguments through **signature-aware delivery**: a hook receives
only the collaborators it declares (or all of them, if it declares
`**kwargs`), so a no-arg `_ainit(self)` or a narrowly-typed override is
never crashed by an undeclared injected collaborator — that collaborator
stays reachable on `self.components`. An override that consumes a
collaborator declares it keyword-only with a default:
`async def _ainit(self, *, dep=None)`. A collaborator parameter *without*
a default (or a required positional) still breaks the zero-injection call
and is rejected by `assert_structured_config_consumer`.

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

### Collaborator injection

The construction contract distinguishes two kinds of input:

- **Config** — scalar knobs and nested sub-configs. Flows through
  `CONFIG_CLS` and lands on `self.config`.
- **Injected collaborators** — *objects* the orchestrating parent
  supplies that are **not** part of this object's own config: a shared
  knowledge base threaded into several strategies, a bot's main LLM
  passed as a memory fallback, a pre-built store handed to a child. In an
  interconnected object graph these are the norm, not the exception.

Collaborators travel through a keyword channel distinct from config and
never enter `self.config`. They land on the read-only `self.components`
mapping and, on the async path, are delivered to `_ainit` as keyword
arguments:

```python
class Strategy(StructuredConfigConsumer[StrategyConfig]):
    CONFIG_CLS: ClassVar[type[StrategyConfig]] = StrategyConfig

    async def _ainit(self, *, knowledge_base=None) -> None:
        # Injected collaborator — supplied by the parent, not in config.
        self._kb = knowledge_base
        # Config-derived collaborator — built from this object's config.
        self._store = await VectorStore.from_config_async(self.config.store)


strategy = await Strategy.from_config_async(
    {"store": {...}},          # config
    knowledge_base=shared_kb,  # injected collaborator
)
assert strategy.components["knowledge_base"] is shared_kb
```

The collaborator-free call sites (`from_config(config)`, `cls(config)`,
`from_config_async(config)`) are unchanged — `self.components` is empty
and `_ainit` receives nothing. Delivery is **signature-aware**: a hook
receives only the collaborators it declares (or all of them, if it
declares `**kwargs`), so a no-arg or narrowly-typed override is never
crashed by an undeclared injected collaborator — it stays reachable on
`self.components`. An `_ainit` (or `_adopt_components`) override that
consumes a collaborator declares it **keyword-only with a default** so
the zero-injection path stays safe; a parameter without a default (or a
required positional) is rejected by `assert_structured_config_consumer`.

#### Dual input: `from_components`

When the parent already holds fully-built collaborators (and so should
*not* have the child rebuild them from config), assemble via
`from_components` instead of `from_config`:

```python
class Bot(StructuredConfigConsumer[BotConfig]):
    CONFIG_CLS: ClassVar[type[BotConfig]] = BotConfig

    def _adopt_components(self, *, llm=None, memory=None) -> None:
        # Bind pre-built collaborators (the config-driven build is skipped).
        self._llm = llm
        self._memory = memory

    async def _ainit(self, *, llm=None, memory=None) -> None:
        if self._prebuilt:
            return  # already wired by from_components — don't rebuild
        self._llm = await build_llm(self.config.llm)
        self._memory = await build_memory(self.config.memory)


# Config-driven build:
bot = await Bot.from_config_async({"llm": {...}, "memory": {...}})

# Pre-built assembly (config is an optional scalar-knob snapshot):
bot = Bot.from_components(llm=prebuilt_llm, memory=prebuilt_memory)
```

`from_components` stores the collaborators on `self.components`, sets
`self._prebuilt = True`, and calls `_adopt_components` so the subclass
binds its collaborator attributes. A consumer's `_ainit` should
short-circuit when `self._prebuilt` is `True`. `config` defaults to
`CONFIG_CLS()` (valid only when every field has a default — pass a
`config` snapshot for configs with required fields; omitting it for a
required-field config raises a clear `ValueError` naming the class and
the remedy, not a cryptic dataclass `TypeError`).

#### Post-construction injection: `set_component`

Construction-time injection covers collaborators that exist when the
consumer is built. Two shapes cannot be construction-time arguments:

- **Circular dependency** — a collaborator that itself holds a reference
  to the fully-built consumer (e.g. a service that calls back into it)
  cannot exist at construction time; the consumer must be built first,
  then the collaborator, then injected back.
- **Lifespan-scoped resource** — a store or service built later at
  application-lifespan or scenario-setup time, after the consumer exists.

`set_component` (and its bulk companion `set_components`) is the supported
write path for these — it writes the private backing store behind the
read-only `components` view:

```python
consumer = Strategy.from_config({"store": {...}})

# Circular dependency: build the service against the consumer, inject back.
service = ClipService(strategy=consumer)
consumer.set_component("clip_service", service)
assert consumer.components["clip_service"] is service

# Bulk form — the shape you would otherwise reach around for as
# `consumer._components.update(...)`:
consumer.set_components({"clip_service": service, "project_store": store})
```

`allow_overwrite` defaults to `True` (inject-or-replace). Pass
`allow_overwrite=False` for inject-only semantics — a re-inject of an
existing key then raises `ValueError`. For `set_components` the
no-overwrite check is **all-or-nothing**: one clashing key aborts the whole
write, so a partial bulk write never leaves the consumer half-wired.
Setting a name listed in `INTERNAL_COMPONENTS` affects only this consumer
and is never forwarded to children.

**Read-once boundary.** Whether the write *reaches* a consuming subsystem
depends on when that subsystem reads its collaborators:

- A consumer that re-reads `components` / `forwardable_components()` afresh
  per operation observes the update on its next read. `WizardReasoning`
  rebuilds its per-stage sub-strategies from config **every turn** (a
  fresh, cacheless `get_registry().create(...)` that reads
  `forwardable_components()` live), so a `set_component` on the wizard
  reaches the **next turn's** sub-strategies. Inject runtime collaborators
  this way — **not** via a post-construction setter on a per-stage *step
  instance*, which is silently discarded before the next turn's rebuild.
- A consumer that reads a collaborator **once** in `_setup` / `_ainit` /
  `_adopt_components` folds it into derived state at construction and does
  **not** re-consume it. For those, call `set_component` before that first
  read. This is impossible for a genuine circular dependency (the
  collaborator does not exist when `_ainit` runs) — such a consumer must
  consume the collaborator lazily per-operation (read `self.components[name]`
  at use-time) rather than binding it in `_ainit`.

#### Declaring required collaborators: `EXPECTED_COMPONENTS`

A consumer that cannot function without a particular injected collaborator
can declare it. `EXPECTED_COMPONENTS` is the read-side counterpart to
`INTERNAL_COMPONENTS`: where `INTERNAL_COMPONENTS` declares *"collaborators
I consume and must not forward to children"*, `EXPECTED_COMPONENTS` declares
*"collaborators I require to be present"*.

```python
class Strategy(StructuredConfigConsumer[StrategyConfig]):
    CONFIG_CLS: ClassVar[type[StrategyConfig]] = StrategyConfig
    EXPECTED_COMPONENTS: ClassVar[frozenset[str]] = frozenset({"clip_service"})
```

The two sets are orthogonal and may overlap or be disjoint. A collaborator
can be **both** (consumed internally *and* required — a composing strategy's
FSM handle it both keeps to itself and cannot run without), **expected-only**
(required but forwarded to children unchanged), or **internal-only** (a cache
the consumer builds itself, required of no injector). Declare each set for its
own concern rather than conflating them.

Three read-side helpers consume the declaration:

```python
# Advertise (classmethod — no instance needed, e.g. for a config lint):
Strategy.expected_components()          # frozenset({"clip_service"})

consumer = Strategy.from_config({...})

# Pure diff — required collaborators not yet present (no raise):
consumer.missing_components()           # frozenset({"clip_service"})

# ... consumer wires the collaborator (construction arg or set_component) ...
consumer.set_component("clip_service", service)

# Loud check — raises ConfigurationError naming any still-missing collaborator:
consumer.require_components()           # returns None once satisfied
```

`missing_components()` returns the set of required-but-absent names rather
than raising, so **the caller picks the policy** — log a warning off the
diff, or call `require_components()` to raise a `ConfigurationError`. Both
default to checking the consumer's own `components`; pass an explicit
`available` iterable to test a candidate set against the class (e.g. a
composing parent checking whether the collaborators it would forward satisfy
a child's requirements).

`require_components()` is **opt-in — never auto-called at construction.** The
mixin does not enforce the declared set from `__init__` / `from_config` /
`from_components` / `_ainit`, because a required collaborator may legitimately
be absent at construction time: a circular dependency injected afterward via
`set_component`, or a lifespan-scoped resource. Call `require_components()` at
*your own* wiring boundary, **after** any `set_component` — for a circular
dependency, after injecting the collaborator, never at construction.

`expected_components()` returns the most-derived class's own
`EXPECTED_COMPONENTS` — there is **no MRO auto-union** (matching
`forwardable_components()` reading `INTERNAL_COMPONENTS`, and the
`CapabilityMixin` convention). A subclass extending the set unions explicitly:

```python
class ExtendedStrategy(Strategy):
    EXPECTED_COMPONENTS: ClassVar[frozenset[str]] = (
        Strategy.EXPECTED_COMPONENTS | {"project_store"}
    )
```

A composing parent (e.g. `WizardReasoning`) forwards collaborators to its
children opaquely via `forwardable_components()` and does **not** yet
auto-enforce a child's declared `EXPECTED_COMPONENTS` — enforcement is the
consumer's own opt-in call.

#### Async registry dispatch: `create_async`

When a registry dispatches polymorphic, asynchronously-constructed
plugins, use `PluginRegistry.create_async` — it awaits the factory
before the `validate_type` guard runs (so the guard checks the resolved
instance, not a coroutine) and forwards `**kwargs` (injected
collaborators) to the factory:

```python
strategy = await registry.create_async(
    config={"strategy": "rag", ...},
    knowledge_base=shared_kb,
)
```

It prefers a `from_config_async` classmethod when present, else awaits an
awaitable `from_config`/factory result; a purely synchronous factory
works unchanged.

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
5. Entry-point symmetry: an overridden `from_config_async` routes
   through `_coerce_config`; an overridden `from_config` routes through
   `_coerce_config` when sync, or delegates to `from_config_async` when
   async (the [async-canonical
   delegator](#async-canonical-construction-the-from_config-async-delegator)).
6. (Optional) A registry factory passed via `expected_factory=`
   delegates to `from_config`.
7. An overridden `_ainit` / `_adopt_components` that names collaborator
   parameters declares them keyword-only with defaults (so the
   zero-collaborator construction path cannot crash).

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

### `assert_polymorphic_bindings_resolve(cls)`

Wiring guard for [`validate()`](#polymorphic-section-validation-validate-config_registries):
asserts every binding a config declares in `_polymorphic_fields` is
registered in `config_registries`. The declaration and the resolver
registration live in different packages, so a rename or dropped
registration would silently turn `validate()` into a skip-and-`debug`
no-op. Run this from each adopting package's test suite (with the owning
packages imported) so a real wiring regression fails in CI:

```python
from dataknobs_common.testing import assert_polymorphic_bindings_resolve

def test_rag_config_bindings_resolve():
    import dataknobs_data.vector.stores  # noqa: F401 — registers the resolver
    assert_polymorphic_bindings_resolve(RAGConfig)
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

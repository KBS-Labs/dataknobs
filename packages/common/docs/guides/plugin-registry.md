# Plugin Registry

The `PluginRegistry` provides a generic, thread-safe registry with factory support for managing plugin-style components. Unlike the base `Registry` (which stores pre-built items), `PluginRegistry` stores factory classes or callables and creates instances on demand, with features tailored to plugin discovery, lazy initialization, and configuration-driven construction.

## Overview

Key capabilities:

- **Factory registration** — Register classes or callables that produce instances on demand
- **Lazy initialization** — Defer built-in registrations until first access via `on_first_access`
- **Configuration-driven lookup** — Extract plugin keys from config dicts automatically
- **Case-insensitive keys** — Optional key canonicalization for user-facing registries
- **Two instantiation modes** — Cached `get()` vs fresh-instance `create()`
- **Type validation** — Ensure registered factories produce the expected type
- **Metadata** — Attach descriptive metadata to registrations

## Quick Start

```python
from dataknobs_common.registry import PluginRegistry


# 1. Define a base type
class Handler:
    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(**config)


# 2. Create a registry with lazy initialization
def _register_builtins(registry):
    registry.register("default", DefaultHandler)
    registry.register("fast", FastHandler)

handlers = PluginRegistry[Handler](
    "handlers",
    validate_type=Handler,
    canonicalize_keys=True,
    config_key="handler_type",
    config_key_default="default",
    on_first_access=_register_builtins,
)

# 3. Create instances from config
handler = handlers.create(config={"handler_type": "fast", "timeout": 5})
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Registry name for identification |
| `default_factory` | `type[T] \| Callable \| None` | `None` | Default factory when key not found |
| `validate_type` | `type \| None` | `None` | Base type to validate registrations against. A class, an ABC, or a `@runtime_checkable` Protocol — including one carrying properties. See [What `validate_type` checks](#what-validate_type-checks). |
| `canonicalize_keys` | `bool` | `False` | Lowercase all keys for case-insensitive lookup |
| `config_key` | `str \| None` | `None` | Field name to extract lookup key from config mappings |
| `config_key_default` | `str \| None` | `None` | Fallback when `config_key` field is absent |
| `strip_config_key` | `bool` | `False` | Remove the key field from config before passing it to the factory, when `key` is not given. An explicit `key=` leaves the config as given |
| `on_first_access` | `Callable \| None` | `None` | Lazy init callback (supports re-entrant `register()` calls) |
| `not_found_kind` | `str \| None` | `None` | Opt-in kind label rendered into the not-found error from `create()` / `create_async()`. When set (e.g. `"event bus backend"`), the message becomes `"Unknown event bus backend: <key>. Available backends: <sorted-keys>"`. When `None`, the historical `"Plugin '<key>' not registered"` text is used. |
| `not_found_exception` | `type[Exception]` | `NotFoundError` | Exception class raised on not-found. Defaults to `NotFoundError` (the `DataknobsError`-rooted shape consumers catch programmatically). Domain shims preserving a historical `ValueError` contract pass `not_found_exception=ValueError`. Non-`DataknobsError` classes receive the message only (no `context=` kwarg). |

## Per-input-shape Split Convention

When using `PluginRegistry` for a Protocol parameterized by input shape (e.g. `ResourceResolver[KeyT, ValueT]`, `Discriminator[InputT, KindT]`), prefer N typed registries (one per concrete input shape) over one flat registry with `validate_type=Any`.

The typed `validate_type=` is load-bearing under consumer-extensibility: an out-of-tree backend that structurally conforms to the wrong Protocol shape would silently register and only fail at use-time without the constraint.

Worked example — generic resolvers (`KeyT → ValueT | None` lookups) and partition resolvers (`record → str | None` lookups) get separate registries:

```python
from dataknobs_common.registry import PluginRegistry
from dataknobs_common.resolver import ResourceResolver

resolver_backends: PluginRegistry[ResourceResolver[Any, Any]] = PluginRegistry(
    name="resolver_backends",
    config_key="backend",
    config_key_default="mapping",
)

partition_resolver_backends: PluginRegistry[Any] = PluginRegistry(
    name="partition_resolver_backends",
    config_key="backend",
    config_key_default="null",
)
```

If a consumer later surfaces "actually we wanted one flat registry," the cost of being wrong is one line per entry (move entries between registries; deprecate the smaller one). The choice is reversible; the typed pin is not.

## What `validate_type` checks

Two checks, at two moments, and a Protocol is a first-class argument to both.

| When | What is checked | On failure |
|---|---|---|
| `register()` / `set_default_factory()`, **class factories only** | the class can produce `validate_type` | `TypeError`, naming the registry, the base, the class, and any missing members |
| `get()` / `create()` and their async twins, **every factory** | the instance the factory returned is a `validate_type` | `OperationError` |

A callable factory's return type is unknowable before it runs, so it is
checked at the second moment only. That asymmetry is inherent, not a gap:
the shape is still caught before the instance reaches a caller.

**Protocols carrying properties.** `issubclass` refuses a Protocol with any
non-method member — the whole call, not the offending member — so the class
check falls back to comparing the class against the Protocol's declared
members. That fallback is the same check `issubclass` performs for a
method-only Protocol, moved to a shape `issubclass` will not accept, so a
property-carrying Protocol and its method-only twin reach the same verdict
on the same class.

**Protocols must be `@runtime_checkable`.** A Protocol without the decorator
supports neither `isinstance` nor `issubclass`, so neither check can run
against it and the registry is inert in both directions. `register()` says
so and names the decorator rather than letting registration pass and
`create()` fail for an unstated cause.

**A member's async-ness is part of the shape.** `@runtime_checkable` asks
whether a member is *present* and nothing more, so a Protocol and its async
twin — the same member name, one `def` and one `async def` — satisfy each
other. Neither `isinstance` nor `issubclass` can separate them, which means
a registry gating on `ResourceResolver` admitted an implementation whose
`resolve` was `async def`, and `create()` handed the caller an object whose
method returns a coroutine. Nothing raised: the value is truthy and not
`None`, and the only trace was a `RuntimeWarning` at interpreter shutdown
naming the implementation rather than the registry.

Both checks in the table above therefore compare each declared member's
async-ness against the candidate's, and refuse a mismatch in either
direction:

> Factory class `AsyncFetcher` cannot satisfy `Fetches`: its `fetch()` is
> asynchronous where `Fetches.fetch()` is not (registry `'fetchers'`,
> plugin `'remote'`).

Properties are skipped — a property carries no async-ness to disagree about
— and so is a member the candidate does not have at all, which is the
presence check's business.

This is asked *inside* the gate rather than fixed by renaming the members,
which has two consequences worth knowing. It covers a twin pair **you**
define, not merely the pairs shipped here. And `isinstance` is untouched:
it still answers `True` in both directions, because that answer belongs to
the language rather than to this registry. A bare `isinstance` against one
half of a twin pair remains a smoke test; `validate_type` is what makes it
a check.

## `BackendRegistry` Protocol

Consumers writing tooling that asks "is this thing a registry-like object?" should `isinstance` against the runtime_checkable `BackendRegistry` Protocol instead of the concrete `Registry` or `PluginRegistry` classes:

```python
from dataknobs_common import BackendRegistry, PluginRegistry, Registry

items = Registry[str]("items")
plugins = PluginRegistry[Any]("plugins")

assert isinstance(items, BackendRegistry)
assert isinstance(plugins, BackendRegistry)
```

`BackendRegistry` is deliberately minimal — it covers the four methods every registry-like adopter must offer: the `name` property, `has(key)`, `list_keys()`, and `unregister(key)`. Members specific to one shape — `create()` / `create_async()` / `get_factory()` for `PluginRegistry`; `get_metrics()` / `list_items()` / `items()` / `count()` / `clear()` for `Registry` — are NOT in the Protocol. Consumers needing those features should `isinstance` against the concrete class.

`BackendRegistry` joins `ResourceResolver`, `Discriminator`, and `CapabilityContract` as the cross-cutting runtime_checkable Protocols re-exported from the top-level `dataknobs_common` namespace.

## `get()` vs `create()`

The registry supports two modes of instantiation with different calling conventions:

| Feature | `get()` | `create()` |
|---------|---------|------------|
| **Caching** | Returns cached instances | Always creates fresh instances |
| **Factory signature** | `factory(key, config)` | `factory.from_config(config, **kwargs)` or `factory(config, **kwargs)` |
| **Key resolution** | Required positional arg | Optional — can extract from config via `config_key` |
| **Async twin** | `get_async()` | `create_async()` |
| **Use case** | Singletons, shared resources | Per-request instances, config-driven construction |

The two factory signatures are the reason the async twins are not
interchangeable: sending a `get()` caller to `create_async()` would change
the arity out from under their factory. Each sync method names its own twin.

### Typed configs

`create()` and `create_async()` take a `StructuredConfig` as well as a
mapping. Their parameter is annotated `PluginConfig`, which is exported from
`dataknobs_common`. That is what the `from_config` / `from_config_async`
constructors they dispatch to accept, and the config reaches the factory
unchanged. Reading the key through `config_key` needs a mapping, so pass
`key=` with a typed config:

```python
from dataclasses import dataclass
from typing import ClassVar

from dataknobs_common import PluginRegistry
from dataknobs_common.structured_config import (
    StructuredConfig,
    StructuredConfigConsumer,
)


@dataclass(frozen=True)
class WidgetConfig(StructuredConfig):
    size: int = 1


class Widget(StructuredConfigConsumer[WidgetConfig]):
    CONFIG_CLS: ClassVar[type[WidgetConfig]] = WidgetConfig


registry: PluginRegistry[Widget] = PluginRegistry("widgets", config_key="type")
registry.register("widget", Widget)

widget = registry.create("widget", config=WidgetConfig(size=4))  # key named
registry.create(config={"type": "widget", "size": 4})  # key read from a mapping
# registry.create(config=WidgetConfig(size=4))  -> TypeError: pass key=
```

Leaving `key` out with a typed config on a `config_key` registry raises
`TypeError`, naming the registry and the key it could not read. That happens
even when `config_key_default` is set. The registry does not fall back to the
default and does not guess the key from an attribute: a typed config's field
is not named after the config key, and falling back would build a plugin the
config did not ask for.

`get()` and `get_async()` take any mapping, but not a typed config. Their
factories are called as `factory(key, config)`, and no factory contract says
what a typed config would mean there.

The key is read from a `collections.abc.Mapping` only. An object that answers
`in` and `[]` without being one is refused the same way as a typed config.

## Asynchronous factories

A factory may be `async def`, or may return any awaitable. Register it the
same way:

```python
async def build_backend(config, **kwargs):
    backend = Backend(config)
    await backend.connect()
    return backend

registry.register("remote", build_backend)

backend = await registry.create_async("remote", config)
```

**Only the async methods can await one.** `get()` and `create()` raise
`OperationError` naming the registry, the key, and the async method to call
instead:

> Plugin `'remote'` in registry `'backends'` has an asynchronous factory
> (`build_backend`); this method cannot await it. Call `create_async()`
> instead.

That refusal is the point rather than a limitation. A synchronous caller
cannot await, so its only honest answers are the instance or an error —
and the third answer, handing back the un-awaited coroutine, produces no
exception, no log line, and a `RuntimeWarning` at interpreter shutdown
attributed to the factory rather than to the registry. Through `get()` it
was worse still: the coroutine was *cached*, so the first caller awaited it
successfully and every later one received the same exhausted object.

A synchronous factory needs no change and works through all four methods.

## Errors from a factory

All four entry points wrap an exception the factory raised, naming the key
and the exception type and carrying the original on `__cause__`. The factory's
own words, and its `context`, stay out: a factory builds a backend from
deployment config, so its failure text is a driver's or an SDK's and can carry
the connection URL it was handed.

| The factory raised | The caller gets |
|---|---|
| a `NotFoundError`, or a subclass | `NotFoundError("Failed to create plugin '<key>' (<type>)")` |
| anything else, `OperationError` and its subclasses included | `OperationError("Failed to create plugin '<key>' (<type>)")` |
| a `NotFoundError` or `OperationError` a registry raised | that error, unchanged |
| a `ConfigFieldError` that `StructuredConfig` wrote, or an error other than a `NotFoundError` raised `from` one | `PluginConfigError("Failed to create plugin '<key>': <refusal>")` |

So a caller catching either base catches what it caught before. One catching
a *subclass* raised inside the factory finds it on `__cause__`. The type is
not enough to exempt an error from the bound, because it says nothing about
who wrote the text: `OperationError` is a public base a factory raises with
its own words.

**A config refusal keeps its words.** A `ConfigFieldError` that
`StructuredConfig.from_dict` raised names the config class, its fields and the
keys the config wrote, and never a value, so the registry carries it: the
caller gets a `PluginConfigError` whose message ends with the refusal's and
whose `context` is the refusal's plus `key` and `registry`. It is an
`OperationError` as well as a `ConfigFieldError`, so `except OperationError`
catches it as before.

```python
sources = PluginRegistry("sources", config_key="type", strip_config_key=True)
sources.register("file", FileSource)   # a StructuredConfigConsumer

sources.create(config={"type": "file", "pth": "data.csv"})
# PluginConfigError: Failed to create plugin 'file': FileSourceConfig does not
#   accept 'pth' (did you mean 'path'?). Accepted keys: path, sheet.
```

What decides is a mark only `StructuredConfig` sets, never the type: a
factory raising its own `ConfigFieldError` is bounded like any other error.
The registry also follows `raise ... from`, so a factory that raises its own
error from a refusal reports the refusal's words and not its own. It does not
follow an error raised merely while handling a refusal, which may be one the
factory recovered from. A `NotFoundError` raised `from` a refusal stays a
bounded `NotFoundError`, so a caller catching that family keeps catching it.

The mark records the refusal's words as `from_dict` wrote them, and those are
what the registry carries. A factory that amends the refusal before raising it
— prefixing an origin with `naming_refusals`, or adding to its `context` —
adds nothing of its own: the amended refusal is still on `__cause__`, for the
logs.

A refusal is reported wherever it is on the `__cause__` chain, including one
about config the factory built for itself rather than the caller's. Its text
still names no value, but it reads as the caller's config error, so a factory
that reads settings of its own with `from_dict` should catch that refusal and
raise its own error rather than raise `from` it.

A `NotFoundError` or `OperationError` a *registry* raised is text this module
wrote, so it passes through an outer factory as itself. A registry whose
`not_found_exception` is outside those families, such as `ValueError`, still
has its miss wrapped in `OperationError`, so an outer call raises only the
types it always has. A composite factory that resolves a sub-component
through another registry and misses raises that registry's `NotFoundError`,
with the inner key named:

```python
def composite(config, **kwargs):
    # resolves a sub-component through another registry
    return Composite(sub_backends.create(config["sub"]))

backends.register("composite", composite)

try:
    backends.create("composite", {"sub": "missing"})
except NotFoundError as e:
    # the inner registry's own error, not "Failed to create plugin"
    e.context["registry"]  # -> "sub_backends"
```

Note the factory's `(config, **kwargs)` signature: that is the `create` lane.
A factory registered for `get()` takes `(key, config)` instead — see
[`get()` vs `create()`](#get-vs-create).

`PluginFactory[T]` is the exported alias for all three accepted shapes —
a class, a callable returning an instance, and a callable returning an
awaitable one — for consumers annotating a factory before registering it:

```python
from dataknobs_common import PluginFactory

def make_backend() -> PluginFactory[Backend]:
    ...
```

## Wrapping a registry in a helper

A package usually puts a function in front of its registry —
`create_memory_from_config(config)` rather than
`memory_backends.create(config=config)` — to turn the registry's miss into
"Unknown memory type: … Available types: …" and to keep the exception types
its callers already catch. The tempting body reads the registry's bounded
error back off `__cause__`:

```python
# Don't: this reopens what the registry closed.
except OperationError as exc:
    if isinstance(exc.__cause__, ValueError):
        raise exc.__cause__ from None
```

The type cannot tell a diagnostic you wrote from a driver's text. "Summary
memory requires an LLM provider" and `ValueError("could not connect to
postgres://admin:pw@db/prod")` are the same type, so the unwrap hands the
second to your caller — and, behind an HTTP boundary that shows validation
messages, to theirs.

`create_surfacing` is that body, written once:

```python
from dataknobs_common import create_surfacing

def create_widget(config: dict[str, Any]) -> Widget:
    return create_surfacing(
        widget_backends,
        subject="widget",
        unknown=lambda e: ValueError(
            f"Unknown widget type: {e.context['key']}. "
            f"Available types: {', '.join(sorted(widget_backends.list_keys()))}"
        ),
        config=config,
    )
```

| When creating through the registry fails with | The helper raises |
|---|---|
| the registry's miss for the key | `unknown(error)`, or the `NotFoundError` when `unknown` is omitted |
| a key declared unavailable (`declare_unavailable`) | the registry's `PluginUnavailableError("Plugin '<key>' is not available here: <reason>")` — both a `NotFoundError` and a `ValueError` |
| a `PluginConfigError` | that error, as built |
| a factory error a registry raised, or dataknobs wrote, and that has not changed since | that error, as itself |
| a factory's `ValueError` or `ValidationError` | `FactoryValueError("Failed to create widget '<key>' (ValueError)")` — both a `ValueError` and a `ValidationError` |
| a factory's `ResourceError`, `TimeoutError`, `ConcurrencyError`, `SerializationError` or `ConfigurationError` | a bounded error of the same family |
| anything else a factory raised | the registry's bounded `OperationError` |

In every bounded case the factory's error is on `__cause__`, for a log.
`unknown` is called only for this registry's own miss, never for a factory's
`NotFoundError`, a miss in another registry a factory looked something up
in, or a key declared unavailable: that name is spelled correctly, and the
registry's reason says what to install. `create_surfacing_async` is the twin
over `create_async`.

A helper that writes its own message around the failure rather than raising
it asks `surfaced_cause(exc)` instead: the cause when its words may be shown,
else `None`, in which case name the cause by `type(exc.__cause__).__name__`.

**Your own factories' words are bounded.** What `surfaced_cause` recognises
is a registry's error and the refusals dataknobs itself raises, marked where
they are raised. A `ValueError` or `ValidationError` your factory raises
reaches your caller as `FactoryValueError` naming the key, which is the safe direction: whatever
your factory says goes to the log, not to whoever is on the other side of
your API.

`assert_no_cause_read` in `dataknobs_common.testing` keeps the unwrap from
coming back: it fails on any read of `__cause__` or `__context__` under the
roots you give it outside an allow-list of reviewed readers, including one
spelled `getattr(exc, "__cause__")` and a `traceback` formatter, which renders
the whole chain.

## Lazy Initialization

The `on_first_access` callback runs once before the first public method call. This is useful for deferring imports and registrations:

```python
def _register_builtins(registry):
    # Deferred imports avoid circular dependencies
    from .handlers import DefaultHandler, FastHandler
    registry.register("default", DefaultHandler)
    registry.register("fast", FastHandler)

handlers = PluginRegistry[Handler](
    "handlers",
    on_first_access=_register_builtins,
)
```

The callback supports re-entrancy — calling `register()` from within the callback works correctly. If the callback raises an exception, the registry resets and retries on next access.

## Usage in DataKnobs

`PluginRegistry` is used as the backing store for several domain registries:

- **Strategy registry** (`dataknobs-bots`) — Reasoning strategy discovery
- **LLM provider registry** (`dataknobs-llm`) — Provider class lookup
- **Database backend registries** (`dataknobs-data`) — Sync and async backend discovery
- **Vector store registry** (`dataknobs-data`) — Vector backend discovery

## Import

```python
from dataknobs_common.registry import PluginRegistry
```

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
| `config_key` | `str \| None` | `None` | Field name to extract lookup key from config dicts |
| `config_key_default` | `str \| None` | `None` | Fallback when `config_key` field is absent |
| `strip_config_key` | `bool` | `False` | Remove key field from config before passing to factory |
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

`PluginFactory[T]` is the exported alias for all three accepted shapes —
a class, a callable returning an instance, and a callable returning an
awaitable one — for consumers annotating a factory before registering it:

```python
from dataknobs_common import PluginFactory

def make_backend() -> PluginFactory[Backend]:
    ...
```

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

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
| `validate_type` | `type \| None` | `None` | Base type to validate registrations against |
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

## `get()` vs `create()`

The registry supports two modes of instantiation with different calling conventions:

| Feature | `get()` | `create()` |
|---------|---------|------------|
| **Caching** | Returns cached instances | Always creates fresh instances |
| **Factory signature** | `factory(key, config)` | `factory.from_config(config, **kwargs)` or `factory(config, **kwargs)` |
| **Key resolution** | Required positional arg | Optional — can extract from config via `config_key` |
| **Use case** | Singletons, shared resources | Per-request instances, config-driven construction |

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

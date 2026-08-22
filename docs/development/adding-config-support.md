# Adding Configuration Support to DataKnobs Packages

This guide provides step-by-step instructions for adding configuration support
to new or existing DataKnobs packages and classes.

## When to Add Configuration Support

Add configuration support when:

- Your class might be instantiated from external configuration files
- Users need to configure your class without modifying code
- Your class is part of a larger system that uses dependency injection
- You want to support environment-based configuration
- Your class has complex initialization parameters

## The Two Pieces

Configuration support is a pair of classes, not one:

- **`StructuredConfig`** — a frozen dataclass holding the knobs. It is the
  schema: the field set defines what the configuration accepts, and
  `from_dict()` / `to_dict()` are derived from it rather than hand-written.
- **`StructuredConfigConsumer[ConfigT]`** — a generic mixin your class inherits.
  It provides one `__init__` accepting a typed config, a dict or loose kwargs;
  a typed `self.config` property; `from_config()` and `from_config_async()`
  entry points; and the `_setup()` / `_ainit()` hooks where your class does its
  own work.

Because the dataclass *is* the construction surface, the two cannot drift: a
field you add is accepted immediately, and a key the schema does not know is
either rejected or reported, never silently dropped into a default.

## Step-by-Step Implementation

### 1. Add Dependencies

Both primitives live in `dataknobs-common`:

```toml
# pyproject.toml
[project]
dependencies = [
    "dataknobs-common>=3.0.0",
    # ... other dependencies
]

[tool.uv.sources]
dataknobs-common = { workspace = true }
```

Add `dataknobs-config` as well only if your package also *loads* configuration
files or builds objects from them — a class that merely accepts a config
dataclass does not need it.

### 2. Define the Config Dataclass

Give every knob a name, a type and a default. Frozen, so a config cannot be
mutated out from under the object holding it:

```python
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from dataknobs_common.structured_config import StructuredConfig


@dataclass(frozen=True)
class MyServiceConfig(StructuredConfig):
    """Configuration for MyService.

    Attributes:
        host: Server hostname.
        port: Server port.
        timeout: Connection timeout in seconds.
        tags: Optional labels attached to every request.
    """

    host: str = "localhost"
    port: int = 8080
    timeout: float = 30.0
    tags: tuple[str, ...] = ()

    _UNKNOWN_KEYS: ClassVar[Literal["ignore", "raise"]] = "raise"
```

`_UNKNOWN_KEYS = "raise"` is the line worth arguing about. The default is
`"ignore"`, which is right when a config travels with routing keys that are not
fields. It is wrong when every field has a working default, because then a
misspelling does not fail — it succeeds against the wrong thing:

```python
MyServiceConfig.from_dict({"hosst": "db.internal"})
# ValueError: MyServiceConfig does not accept 'hosst' (did you mean 'host'?).
#             Accepted keys: host, port, tags, timeout.
```

Use a mutable default through `field(default_factory=...)`, exactly as in any
dataclass.

### 3. Make the Class a Consumer

Name the config class in `CONFIG_CLS`, and put derived state in `_setup()`:

```python
from typing import ClassVar

from dataknobs_common.structured_config import StructuredConfigConsumer


class MyService(StructuredConfigConsumer[MyServiceConfig]):
    """A configurable service."""

    CONFIG_CLS: ClassVar[type[MyServiceConfig]] = MyServiceConfig

    def _setup(self) -> None:
        """Initialize derived attributes computed from ``self.config``."""
        self.endpoint = f"http://{self.config.host}:{self.config.port}"
```

That is the whole adoption. Four construction shapes now reach the same state,
and you wrote none of them:

```python
MyService()                                        # all defaults
MyService(MyServiceConfig(host="db", port=5432))   # typed
MyService({"host": "db", "port": 5432})            # a loaded dict
MyService(host="db", port=5432)                    # loose kwargs
MyService.from_config({"host": "db", "port": 5432})  # the registry path
```

`from_config` is the entry point to prefer programmatically, and it is the one
`Config.build_object` calls — see [Integration Tests](#3-integration-tests).

`_setup()` is for derived attributes, not for normalizing input. Normalization
belongs in the config dataclass, which is the next step.

### 4. Normalizing Input That Does Not Match the Field Set

When the incoming dict's shape differs from the fields — a legacy key, an alias,
a value assembled from several others — override `_normalize_dict` on the config
class. It receives a shallow copy and may mutate it freely:

```python
@dataclass(frozen=True)
class MyServiceConfig(StructuredConfig):
    host: str = "localhost"

    _UNKNOWN_KEYS: ClassVar[Literal["ignore", "raise"]] = "raise"
    _INPUT_KEYS: ClassVar[frozenset[str]] = frozenset({"hostname"})

    @classmethod
    def _normalize_dict(cls, raw: dict) -> dict:
        if "hostname" in raw:
            raw.setdefault("host", raw.pop("hostname"))
        return raw
```

Two rules go with a `"raise"` class. Declare the spellings you accept but do not
keep in `_INPUT_KEYS`, so the rejection message can list them; and **remove every
input key you consume**, since the unknown-key check runs on what
`_normalize_dict` returns.

### 5. Nested and Composed Configuration

A field typed as another `StructuredConfig` is recursed into by `from_dict`, so
a nested mapping becomes a nested config object without any code of yours:

```python
@dataclass(frozen=True)
class CacheConfig(StructuredConfig):
    size: int = 1000
    ttl: int = 3600


@dataclass(frozen=True)
class ServerConfig(StructuredConfig):
    host: str = "localhost"
    cache: CacheConfig = field(default_factory=CacheConfig)


config = ServerConfig.from_dict({"host": "db", "cache": {"size": 50}})
config.cache.ttl  # 3600 — the nested default survived
```

### 6. Async Setup

Work that must be awaited goes in `_ainit()`, which runs after `_setup()` and
only on the async path:

```python
class MyService(StructuredConfigConsumer[MyServiceConfig]):
    CONFIG_CLS: ClassVar[type[MyServiceConfig]] = MyServiceConfig

    def _setup(self) -> None:
        self._connection = None

    async def _ainit(self) -> None:
        self._connection = await open_connection(self.config.host)


service = await MyService.from_config_async({"host": "db"})
```

The synchronous `__init__` / `from_config` path does not run `_ainit`, so a
class needing it must be constructed through `from_config_async`.

### 7. Implement Factory Pattern (Optional)

For creating different implementations based on configuration:

```python
from typing import Any

from dataknobs_config import FactoryBase


class MyServiceFactory(FactoryBase):
    """Factory for creating service instances based on type."""

    def create(self, **config: Any) -> Any:
        """Create an instance based on configuration.

        Args:
            **config: Configuration including a 'type' field.

        Returns:
            An instance of the appropriate class.
        """
        which = config.pop("type", "default")

        if which == "advanced":
            from .advanced import AdvancedService
            return AdvancedService.from_config(config)
        if which == "simple":
            from .simple import SimpleService
            return SimpleService.from_config(config)
        from .default import DefaultService
        return DefaultService.from_config(config)
```

## Testing Configuration Support

### 1. Pin the Pattern

`assert_structured_config_consumer` checks the adoption itself — that
`CONFIG_CLS` is declared and is a `StructuredConfig`, that its field set matches
the constructor's parameters, and that the mixin precedes any other base that
defines a competing `__init__`. One line, and it catches the drift that a
hand-written `from_config` used to let through:

```python
from dataknobs_common.testing import assert_structured_config_consumer


def test_my_service_follows_the_pattern():
    assert_structured_config_consumer(MyService)
```

### 2. Unit Tests

```python
import dataclasses

import pytest


class TestConfigSupport:
    """Test configuration support for MyService."""

    def test_defaults(self):
        """Every field has a default, so a bare construction works."""
        service = MyService()
        assert service.config.port == 8080
        assert service.endpoint == "http://localhost:8080"

    def test_from_config(self):
        """The registry entry point takes a plain mapping."""
        service = MyService.from_config({"host": "db.internal", "port": 5432})
        assert service.config.host == "db.internal"
        assert service.endpoint == "http://db.internal:5432"

    def test_typed_construction(self):
        """A typed config reaches the same state as the mapping did."""
        service = MyService(MyServiceConfig(host="db.internal", port=5432))
        assert service.endpoint == "http://db.internal:5432"

    def test_misspelled_key_is_rejected(self):
        """The schema is what makes this a failure rather than a default."""
        with pytest.raises(ValueError, match="does not accept 'hosst'"):
            MyService.from_config({"hosst": "db.internal"})

    def test_config_is_read_only(self):
        """The config is frozen, so it cannot be mutated after construction."""
        service = MyService()
        with pytest.raises(dataclasses.FrozenInstanceError):
            service.config.host = "elsewhere"
```

### 3. Integration Tests

`Config.build_object` calls `from_config(config)` on the target class when it
has one, and falls back to `cls(**config)` otherwise — it dispatches on the
method, not on a base class. A `StructuredConfigConsumer` therefore works
through the configuration system with nothing further to declare:

```python
from dataknobs_config import Config


def test_config_integration():
    """Test integration with the Config class."""
    config = Config()
    config.load({
        "my_services": [{
            "name": "test_service",
            "class": "mypackage.MyService",
            "host": "test.internal",
            "port": 9000,
        }]
    })

    service = config.get_instance("my_services", "test_service")
    assert isinstance(service, MyService)
    assert service.config.host == "test.internal"
    assert service.config.port == 9000


def test_environment_variables(monkeypatch):
    """Test environment variable substitution."""
    monkeypatch.setenv("MY_HOST", "env.internal")

    config = Config()
    config.load({
        "my_services": [{
            "name": "env_test",
            "class": "mypackage.MyService",
            "host": "${MY_HOST}",
        }]
    })

    service = config.get_instance("my_services", "env_test")
    assert service.config.host == "env.internal"
```

The object-graph layer's own vocabulary — `name`, `class`, `factory`, `type`,
`backend` — is stripped before construction and tolerated under `"raise"`, so a
config entry carrying it does not have to be declared in `_INPUT_KEYS`.

## Documentation Requirements

### 1. Class Docstring

The config dataclass carries the option list, because it is the thing that
defines it. Document the fields there and leave the consumer's docstring to say
what the class *does*:

```python
@dataclass(frozen=True)
class WellDocumentedConfig(StructuredConfig):
    """Configuration for WellDocumentedService.

    Attributes:
        host: Server hostname.
        port: Server port.
        timeout: Connection timeout in seconds.
        ssl: Whether to enable TLS.
        api_key: Credential for the upstream service. Masked in ``repr``.
    """

    host: str = "localhost"
    port: int = 8080
    timeout: int = 30
    ssl: bool = False
    api_key: str | None = None

    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset({"api_key"})


class WellDocumentedService(StructuredConfigConsumer[WellDocumentedConfig]):
    """A service that can be built from a configuration file.

    Example:
        >>> from dataknobs_config import Config
        >>> config = Config("config.yaml")
        >>> service = config.get_instance("services", "production")
    """

    CONFIG_CLS: ClassVar[type[WellDocumentedConfig]] = WellDocumentedConfig
```

### 2. README Examples

Add configuration examples to your package README:

````markdown
## Configuration Support

This package supports the DataKnobs configuration system. Its classes take
their configuration through `StructuredConfigConsumer`, so they can be built
from configuration files as well as constructed directly.

### Example Configuration

```yaml
# config.yaml
my_services:
  - name: processor
    class: mypackage.DataProcessor
    input_dir: /data/input
    output_dir: /data/output
    batch_size: 100

  - name: validator
    class: mypackage.DataValidator
    rules_file: /config/rules.yaml
    strict_mode: true
```

### Loading from Configuration

```python
from dataknobs_config import Config

config = Config("config.yaml")
processor = config.get_instance("my_services", "processor")
validator = config.get_instance("my_services", "validator")
```
````

## Common Patterns

### Pattern 1: Optional Dependencies

Select the optional path from a field, and fail with an actionable message:

```python
class OptionalDependencyService(StructuredConfigConsumer[BackendConfig]):
    CONFIG_CLS: ClassVar[type[BackendConfig]] = BackendConfig

    def _setup(self) -> None:
        if self.config.backend == "advanced":
            try:
                import advanced_library  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "The 'advanced' backend requires 'advanced_library'. "
                    "Install with: pip install mypackage[advanced]"
                ) from exc
            self._setup_advanced()
        else:
            self._setup_basic()
```

### Pattern 2: Validation

Per-class invariants belong in the config's `__post_init__`, where they run for
every construction shape rather than only the one that goes through a dict:

```python
@dataclass(frozen=True)
class ValidatedConfig(StructuredConfig):
    host: str = "localhost"
    port: int = 8080

    _UNKNOWN_KEYS: ClassVar[Literal["ignore", "raise"]] = "raise"

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("'host' must be non-empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("'port' must be between 1 and 65535")
```

Cross-field checks go here too, which is what a per-key validator could never
express.

### Pattern 3: Secrets

List credential-bearing fields in `_SENSITIVE_FIELDS` and redaction is
automatic — the field is masked in `repr`, so it cannot reach a log by
accident:

```python
@dataclass(frozen=True)
class CredentialedConfig(StructuredConfig):
    host: str = "localhost"
    api_key: str | None = None

    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset({"api_key"})


repr(CredentialedConfig(host="db", api_key="sk-live-123"))
# "CredentialedConfig(host='db', api_key='***')"
```

### Pattern 4: Lazy Initialization

```python
class LazyService(StructuredConfigConsumer[MyServiceConfig]):
    CONFIG_CLS: ClassVar[type[MyServiceConfig]] = MyServiceConfig

    def _setup(self) -> None:
        self._connection = None

    @property
    def connection(self):
        """Lazily initialize the connection when first accessed."""
        if self._connection is None:
            self._connection = create_connection(
                self.config.host, self.config.port
            )
        return self._connection
```

Prefer `_ainit()` over lazy properties when the setup is awaitable — a lazy
property cannot await.

## Checklist

Before considering configuration support complete:

- [ ] A frozen `StructuredConfig` dataclass names every knob, with types and defaults
- [ ] The class inherits `StructuredConfigConsumer[ConfigT]` and declares `CONFIG_CLS`
- [ ] `_UNKNOWN_KEYS = "raise"` is set, or its absence is a deliberate choice
- [ ] Derived state is built in `_setup()`; awaitable setup is in `_ainit()`
- [ ] Input shapes that differ from the field set are handled in `_normalize_dict`
- [ ] Credential-bearing fields are listed in `_SENSITIVE_FIELDS`
- [ ] Invariants are enforced in `__post_init__`
- [ ] `assert_structured_config_consumer` pins the adoption
- [ ] Configuration options are documented in the config dataclass docstring
- [ ] Integration tests verify `Config.get_instance()` builds the class
- [ ] Environment variable substitution is tested
- [ ] README includes configuration examples

## Troubleshooting

### Issue: ImportError when using Config.get_instance()

**Solution**: Ensure the module path in the `class` attribute is correct and the
module is importable:

```python
# Correct: full module path
"class": "mypackage.submodule.MyClass"

# Incorrect: missing package prefix
"class": "submodule.MyClass"
```

### Issue: ValueError saying the config does not accept a key

The class opted into `_UNKNOWN_KEYS = "raise"` and the key matches no field.
This is the check working. Either the key is a misspelling — the message
suggests the nearest field — or it is an alias the class means to accept, in
which case translate it in `_normalize_dict` and declare it in `_INPUT_KEYS`.

### Issue: A key is accepted but ignored

The opposite symptom, and it means `_UNKNOWN_KEYS` is still at its `"ignore"`
default. See [Define the Config Dataclass](#2-define-the-config-dataclass).

### Issue: `_ainit` never runs

It runs only on the `from_config_async` path. A class constructed with
`MyService(...)` or `MyService.from_config(...)` gets `_setup()` and nothing
else.

### Issue: Circular imports

**Solution**: Use lazy imports inside methods:

```python
def _setup(self) -> None:
    # Import here to avoid a circular dependency
    from .other_module import OtherClass
    self.component = OtherClass.from_config(self.config.component)
```

## Migrating from ConfigurableBase

> `dataknobs_config.ConfigurableBase` is the deprecated predecessor of this
> pattern. It still works and raises no runtime warning, so the transition can
> be taken a class at a time — but new code should not adopt it, and removal is
> scheduled for a future release. See
> [ConfigurableBase (deprecated)](../packages/config/configurable-base.md) for
> the full rationale.

The move is not a rename. `ConfigurableBase.from_config` splats the mapping into
the constructor (`cls(**config)`), so the constructor signature is the schema and
nothing checks a key against it. Adopting the successor means introducing the
schema that was previously implicit:

1. Write a `StructuredConfig` dataclass whose fields are the constructor's
   keyword arguments, with the same defaults.
2. Change the base to `StructuredConfigConsumer[YourConfig]` and declare
   `CONFIG_CLS`.
3. Move the constructor body — everything after the parameters were assigned —
   into `_setup()`, reading `self.config.*` instead of the parameters.
4. Delete the hand-written `from_config`; the mixin provides one.
5. Add `assert_structured_config_consumer(YourClass)` to the tests, which will
   fail if step 1 missed a parameter.

Callers need no change: `from_config({...})` and `Config.get_instance(...)` are
the same calls before and after.

## Next Steps

1. Review the [Configuration System Documentation](./configuration-system.md)
2. Read [Structured Configuration](../packages/common/structured-config.md) for
   the full API, including collaborator injection and polymorphic sections
3. Look at the backends in `dataknobs_data`, which all use this pattern
4. Add your class to the package documentation

# Configuration Inheritance

The `InheritableConfigLoader` provides simple YAML/JSON configuration loading with inheritance support via an `extends` field. This complements the main `Config` system for scenarios where you need lightweight, single-file configuration loading with inheritance chains.

## Overview

When building applications with multiple environments or domains, you often have:
- A base configuration with common settings
- Environment-specific overrides (dev, staging, prod)
- Domain-specific configurations that inherit from a common base

The `InheritableConfigLoader` handles this pattern elegantly.

## Quick Start

```python
from dataknobs_config import InheritableConfigLoader

# Create loader
loader = InheritableConfigLoader("./configs")

# Load configuration (resolves inheritance automatically)
config = loader.load("production")
```

Or use the convenience function:

```python
from dataknobs_config import load_config_with_inheritance

config = load_config_with_inheritance("configs/production.yaml")
```

## Configuration Files

### Base Configuration

```yaml
# configs/base.yaml
llm:
  provider: openai
  model: gpt-4
  temperature: 0.7

knowledge_base:
  chunk_size: 500
  overlap: 50

logging:
  level: INFO
```

### Child Configuration

```yaml
# configs/production.yaml
extends: base

llm:
  model: gpt-4-turbo
  api_key: ${OPENAI_API_KEY}

logging:
  level: WARNING
```

When you load `production`, the loader:
1. Loads `base.yaml`
2. Deep merges `production.yaml` on top
3. Substitutes environment variables
4. Returns the merged configuration

Result:
```python
{
    "llm": {
        "provider": "openai",        # From base
        "model": "gpt-4-turbo",      # Overridden
        "temperature": 0.7,          # From base
        "api_key": "sk-..."          # From env var
    },
    "knowledge_base": {              # From base
        "chunk_size": 500,
        "overlap": 50
    },
    "logging": {
        "level": "WARNING"           # Overridden
    }
}
```

## Deep Merge Behavior

Child values override parent values at the deepest level:

```python
from dataknobs_config import deep_merge

base = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "pool": {"min": 1, "max": 10}
    }
}

override = {
    "database": {
        "host": "prod.db.com",
        "pool": {"max": 50}
    }
}

result = deep_merge(base, override)
# {
#     "database": {
#         "host": "prod.db.com",  # Overridden
#         "port": 5432,           # Preserved
#         "pool": {
#             "min": 1,           # Preserved
#             "max": 50           # Overridden
#         }
#     }
# }
```

**Important**: Lists are replaced entirely, not merged:

```python
base = {"items": [1, 2, 3]}
override = {"items": [4, 5]}
result = deep_merge(base, override)
# {"items": [4, 5]}
```

## Multi-Level Inheritance

Configurations can chain inheritance:

```yaml
# configs/base.yaml
app:
  name: MyApp
  version: 1.0

# configs/development.yaml
extends: base

app:
  debug: true

database:
  host: localhost

# configs/local.yaml
extends: development

database:
  host: 127.0.0.1
  name: local_db
```

Loading `local`:
```python
config = loader.load("local")
# Resolves: base -> development -> local
```

## Environment Variable Substitution

### Required Variables

```yaml
database:
  password: ${DB_PASSWORD}  # Raises error if not set
```

### Default Values

```yaml
database:
  host: ${DB_HOST:localhost}  # Uses "localhost" if not set
  port: ${DB_PORT:5432}
```

### Path Expansion

Tilde paths are expanded after substitution:

```yaml
paths:
  data_dir: ${DATA_DIR:~/data}  # Expands ~ to home directory
```

### Disabling Substitution

```python
# Load without environment variable substitution
config = loader.load("config", substitute_vars=False)
```

## Name Resolution

By default a configuration name is a file name directly under `config_dir`.
`resolve_name()` is the seam for changing that, and it applies to every
`extends:` target as well as to the config you asked for — which is what
makes it useful. Parents are named *inside* config files, and the recursion
into them happens inside the loader, so intercepting only the entry point
cannot express a layout convention.

Given a tree whose children name their parents bare:

```
configs/
  domains/
    parent.yaml      # a: 1
    child.yaml       # extends: parent
```

`load("domains/child")` fails without a resolver — the loader looks for
`parent.yaml` at the top level, not beside the child.

### Injecting a resolver

`ResourceResolver` implementations ship in `dataknobs_common`, so neither
common convention needs a class of your own:

```python
from dataknobs_common import CallableResolver, MappingResolver
from dataknobs_config import InheritableConfigLoader

# A layout rule
loader = InheritableConfigLoader(
    "./configs", resolver=CallableResolver(lambda n: f"domains/{n}")
)
loader.load("child")   # {"a": 1, ...} -- `extends: parent` resolved too

# Or a table of aliases
loader = InheritableConfigLoader(
    "./configs", resolver=MappingResolver({"tutor": "domains/bio-tutor"})
)
```

A resolver returning `None` means "no mapping" and falls back to identity,
so a partial mapping leaves every other name alone.

### Overriding the method

When the mapping needs loader state, override the public method instead:

```python
class DomainAwareLoader(InheritableConfigLoader):
    def resolve_name(self, name: str) -> str:
        return f"{self.domain_root}/{name}"
```

The two modes are **alternatives, not layers.** An override replaces the
default implementation, so a loader given both ignores the injected resolver:

```python
class DomainAwareLoader(InheritableConfigLoader):
    def resolve_name(self, name: str) -> str:
        return f"domains/{name}"

# The resolver here is dead code -- the override never consults it.
loader = DomainAwareLoader("./configs", resolver=MappingResolver({"tutor": "x"}))
loader.resolve_name("tutor")   # "domains/tutor", not "x"
```

Constructing that combination warns, because the outcome is otherwise silent
— the loader reads a file you did not configure and says nothing. It is a
warning rather than an error: overriding to normalize or log *and* delegating
to `super()` is a legitimate use of both.

Delegating to `super()` is the other half of the same trap: both mappings then
apply in sequence, and a prefixing pair looks for `domains/domains/child.yaml`.
Pick one mode.

### Enumeration is the other half of the mapping

A resolver answers *"where does this name live"*. Nothing runs it backwards,
so the loader cannot recover the names from the locations — which means
`available_names()` does not follow from `resolve_name()` and has to be
overridden alongside it:

```python
class DomainLoader(InheritableConfigLoader):
    def resolve_name(self, name: str) -> str:
        return f"domains/{name}"

    def available_names(self) -> list[str]:
        return sorted(
            path.stem
            for path in (self.config_dir / "domains").glob("*.yaml")
            if path.is_file()
        )
```

The default is the stems of the files directly under `config_dir`, which is
the set of loadable names only while `resolve_name` is identity. Leaving it
alone under a resolver does not raise — it reports the wrong thing quietly.
Under the layout above every config is a directory down, so the default
returns `[]` and the natural loop runs zero times:

```python
for name in loader.available_names():   # zero iterations, no error
    loader.load(name)
```

A layout that mixes depths is worse than empty: the stems the default does
find are *locations*, and mapping a location through `resolve_name` addresses
something else again. `list_available()` delegates here, so an override
reaches both callers.

### What resolution does not cover

`load_from_file` suppresses resolution for the file **and** its `extends:`
subtree. A mapping is defined relative to `config_dir`, and that method
rebinds `config_dir` to the file's own directory, so applying a convention
there would look for its location beneath a directory you did not choose:

```python
# Resolution off for this whole tree -- `extends:` targets are found
# beside the file, which is what "loads directly from the path" means.
loader.load_from_file("./somewhere-else/svc.yaml")
```

`clear_cache` is the mirror case: it resolves the name you give it, exactly as
`load` does, so **pass the name you passed `load`**. Handing it an
already-resolved name maps it a second time — harmless for a lookup table,
which leaves an unknown name alone, but a prefixing resolver double-prefixes
and the call clears nothing. Nothing is raised; the log line names what was
cleared.

## API Reference

### InheritableConfigLoader

```python
class InheritableConfigLoader:
    def __init__(
        self,
        config_dir: str | Path | None = None,
        *,
        resolver: ResourceResolver[str, str] | None = None,
    ):
        """Initialize loader.

        Args:
            config_dir: Directory containing configs (default: ./configs)
            resolver: Optional name->location mapping (see Name Resolution)
        """

    def resolve_name(self, name: str) -> str:
        """Map a config name to a name/path relative to config_dir.

        Applied to the requested config AND every `extends:` target.
        Default: consult `resolver`, falling back to identity.
        Not applied under `load_from_file`.
        """

    def available_names(self) -> list[str]:
        """The names `load` accepts, for this deployment's layout.

        Default: stems of the files directly under config_dir, which is
        the loadable set only while `resolve_name` is identity. The
        mapping is one-way, so override this alongside it.
        """

    def load(
        self,
        name: str,
        use_cache: bool = True,
        substitute_vars: bool = True,
    ) -> dict[str, Any]:
        """Load configuration with inheritance.

        Args:
            name: Config name without extension
            use_cache: Use cached config if available
            substitute_vars: Substitute environment variables

        Returns:
            Resolved configuration dictionary

        Raises:
            InheritanceError: If config not found or cycle detected
        """

    def load_from_file(
        self,
        filepath: str | Path,
        substitute_vars: bool = True,
    ) -> dict[str, Any]:
        """Load from specific file path.

        Inheritance is resolved relative to the file's directory, and
        `resolve_name` is suppressed for the whole subtree.
        """

    def list_available(self) -> list[str]:
        """List all available configuration names.

        Delegates to `available_names`, which is the one to override.
        """

    def validate(self, name: str) -> tuple[bool, str | None]:
        """Validate a configuration.

        Returns:
            Tuple of (is_valid, error_message)
        """

    def clear_cache(self, name: str | None = None) -> None:
        """Clear configuration cache.

        Pass the name you passed `load` -- this resolves it the same way.
        """
```

### Utility Functions

```python
def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries.

    Override values take precedence. Nested dicts are merged recursively;
    all other types are replaced.
    """

def substitute_env_vars(data: Any) -> Any:
    """Recursively substitute environment variables.

    Supports ${VAR} and ${VAR:default} patterns.
    Expands ~ in paths after substitution.

    Raises:
        ValueError: If required variable not set
    """

def load_config_with_inheritance(
    filepath: str | Path,
    substitute_vars: bool = True,
) -> dict[str, Any]:
    """Convenience function to load a config file with inheritance."""
```

### InheritanceError

```python
class InheritanceError(Exception):
    """Error during configuration inheritance resolution.

    Raised for:
    - Config file not found
    - Circular inheritance detected
    - Invalid YAML/JSON
    - Config is not a dictionary
    """
```

## Caching

Configurations are cached by default for performance:

```python
# First load - reads from disk
config1 = loader.load("production")

# Second load - returns cached version
config2 = loader.load("production")  # Same object

# Read fresh without touching the cache: this neither reads the stored
# entry nor replaces it, so later load() calls still get the cached one
config3 = loader.load("production", use_cache=False)

# Refresh what everyone else sees: evict, then load
loader.clear_cache("production")
config4 = loader.load("production")

# Clear all cache
loader.clear_cache()
```

`use_cache=False` is a bypass, not a refresh. It takes no part in the cache
in either direction — a bypassing load that also wrote would let
`load_from_file`, which reads with `config_dir` rebound to another
directory, store content under a bare name that then answers reads for the
configured one. To make a reload visible to subsequent callers, clear the
entry and load normally.

### Clearing a config clears what inherited from it

`clear_cache(name)` also evicts every config that reached `name` through
`extends:`, transitively. A cached child holds its parent's content merged
in, so clearing only the parent would leave that copy answering with content
the parent no longer has — the staleness the call was made to resolve,
surviving the call. Invalidation runs down the inheritance edges, never up:
clearing a child leaves its parent cached.

### Substitution mode is part of the cache key

Resolving `extends:` loads the parent **without** substitution and expands
the merged result once, at the end — so the same config can be produced in
two forms. The cache keys on both the name and the substitution mode, so an
entry stored by the inheritance recursion can never serve a request that
asked for expansion, or the reverse:

```python
loader.load("child")             # also caches `parent`, unexpanded
loader.load("parent")            # reads from disk and expands -- not the
                                 # unexpanded entry the recursion stored
```

Without this, `load("parent")` would return raw `${VAR}` placeholders after
a child had been loaded, and `load("child")` would expand the parent's
values a *second* time if the parent had been loaded first. Substitution is
not idempotent — a value whose own text contains `${...}` is re-read as a
template on the second pass — so a config's value depended on load order.
See
[Substitution runs once per source](environment-variables.md#substitution-runs-once-per-source).

`clear_cache(name)` clears every variant stored under that name.

### The name half of the key is the resolved name

Everything the loader keys on a name keys on the one that came out of
[`resolve_name`](#name-resolution) — the cache, the cycle-detection set, the
`extends:` invalidation edges, and `clear_cache`. Two spellings of one config
are therefore one entry, one node, and one thing to clear:

```python
loader = InheritableConfigLoader(
    "./configs",
    resolver=MappingResolver({
        "tutor": "domains/child",
        "child": "domains/child",
        "parent": "domains/parent",   # `extends:` targets need mapping too
    }),
)
loader.load("tutor") is loader.load("child")   # True -- one entry
loader.clear_cache("tutor")                    # clears what load("child") stored
```

A table-style resolver has to cover the `extends:` targets as well as the
entry points — an unmapped name falls back to identity, so a bare `parent`
would be looked for at the top level. `CallableResolver` with a layout rule
sidesteps that, since the rule applies to every name.

Splitting these would be worse than the duplication it replaces: a config
that is two things to the cache and one to the cycle detector, or a
`clear_cache("child")` that does not clear what `load("child")` cached.

`load_from_file` bypasses resolution (above) — but it also bypasses the cache
entirely, so it never stores an entry under an unresolved name.

## Error Handling

### Missing Configuration

```python
try:
    config = loader.load("nonexistent")
except InheritanceError as e:
    print(f"Config not found: {e}")
```

### Circular Inheritance

```yaml
# configs/a.yaml
extends: b

# configs/b.yaml
extends: a  # Circular!
```

```python
try:
    config = loader.load("a")
except InheritanceError as e:
    print(f"Circular inheritance: {e}")
```

### Missing Environment Variable

```python
try:
    config = loader.load("config")  # Has ${REQUIRED_VAR}
except ValueError as e:
    print(f"Missing env var: {e}")
```

## Best Practices

1. **Keep Base Minimal**: Put only truly common values in base configs
2. **Use Descriptive Names**: `production.yaml`, `development.yaml`, not `prod.yaml`
3. **Document Required Variables**: Comment which env vars must be set
4. **Validate in CI**: Use `loader.validate()` in tests
5. **Avoid Deep Inheritance**: 2-3 levels maximum for maintainability

## Comparison with Config Class

| Feature | InheritableConfigLoader | Config |
|---------|------------------------|--------|
| **Use Case** | Simple YAML/JSON loading | Complex, type-organized configs |
| **Inheritance** | Single `extends` field | File references with `@` |
| **Structure** | Free-form dictionary | Type-organized arrays |
| **Env Vars** | `${VAR:default}` | `DATAKNOBS_*` pattern |
| **Object Building** | No | Yes (factories, classes) |
| **References** | No | Yes (`xref:type[name]`) |

Choose `InheritableConfigLoader` for simpler configuration needs where you don't need object construction or cross-references.

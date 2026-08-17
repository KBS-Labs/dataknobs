# Environment-Aware Configuration

The `dataknobs-config` package provides an **environment-aware configuration system** for deploying the same application across different environments (development, staging, production) where infrastructure differs.

## Overview

### The Problem

Traditional configurations contain environment-specific details that cause issues when stored in databases or shared registries:

```yaml
# PROBLEMATIC: This config is not portable
database:
  backend: postgres
  connection_string: postgresql://localhost:5432/mydb  # Only works locally

vector_store:
  backend: faiss
  path: ~/data/vectors  # Local path doesn't exist in production
```

When these configs are stored and then loaded in production:
- Localhost URLs don't exist in production containers
- Local file paths don't exist in containerized environments
- Different environments need different backends

### The Solution

Use **logical resource references** (`$resource`) to separate behavior from infrastructure:

```yaml
# PORTABLE: This config works in any environment
bot:
  database:
    $resource: default
    type: databases

  vector_store:
    $resource: knowledge
    type: vector_stores
```

The logical names are resolved at **instantiation time** against environment-specific bindings:

```yaml
# config/environments/development.yaml
resources:
  databases:
    default:
      backend: sqlite
      path: ~/.local/share/myapp/dev.db

# config/environments/production.yaml
resources:
  databases:
    default:
      backend: postgres
      connection_string: ${DATABASE_URL}
```

## Key Concepts

### 1. Late Binding

Environment-specific values are resolved at the **latest possible moment**:

- **Config loading**: Keep placeholders intact (`${DATABASE_URL}`)
- **Config storage**: Store unresolved app config
- **Object instantiation**: Resolve environment bindings

### 2. Logical Resource References

Instead of hardcoded infrastructure, use logical names that map to environment-specific implementations:

```yaml
# In app config
database:
  $resource: conversations
  type: databases
  pool_size: 10  # Merged with resolved config
```

### 3. Capability Requirements

Resource references can declare required capabilities using `$requires`:

```yaml
llm:
  $resource: default
  type: llm_providers
  $requires: [function_calling]   # Validated against resource capabilities
```

`$requires` makes two claims at once, and both are enforced:

- **The resource must exist.** A reference declaring `$requires` against a
  resource this environment does not define raises `ResourceNotFoundError` —
  a resource that is absent satisfies no capability at all. Declare
  `$required: false` alongside it to say "if it is there it must do X; it may
  be absent", which degrades to the reference's inline defaults instead.
- **If it declares `capabilities` metadata, they must cover the
  requirement.** A found resource missing a required capability raises
  `ConfigError` at resolution time. This check also runs on the degraded
  config when `$required: false` opted out of the first claim — "it may be
  absent" is not "and anything will do".

Environment configs declare resource capabilities as metadata:

```yaml
resources:
  llm_providers:
    default:
      provider: ollama
      model: qwen3:8b
      capabilities: [chat, function_calling, streaming]
```

The `$requires` field is stripped during resolution — it is validation
metadata and never reaches the factory.

`capabilities` is **not** stripped. It is read to validate `$requires` and
then passed through with the rest of the resource config, so a factory that
receives one must tolerate the keyword.

### 4. Missing Resources

A reference naming a resource the current environment does not define
resolves to the reference's inline defaults, with a warning — or to `{}` when
it declares none. That is frequently what you want in development and rarely
what you want in production: an empty config handed to a factory generally
does not fail, it produces the factory's default. A degraded
`conversation_storage` binding becomes an in-memory database, which holds
state perfectly until the process restarts.

A reference can therefore declare that its resource must exist:

```yaml
conversation_storage:
  $resource: conversations
  type: databases
  $required: true      # absent in this environment -> raise, do not degrade
```

#### Precedence

Four levels, each owned by someone different, most specific first. Every
level is *unset-means-defer*, so "explicitly false" and "unspecified" stay
distinguishable:

| # | Level | Spelling | Owner |
|---|---|---|---|
| 1 | The reference | `$required: true` on the reference block | the config author |
| 2 | The reference | a non-empty `$requires` (see above) | the config author |
| 3 | Code | `resolve_for_build(strict_resources=True)`, or `EnvironmentAwareConfig(..., strict_resources=True)` | the calling code / embedding app |
| 4 | The environment | `settings: {strict_resources: true}` | the operator |
| — | Default | lenient — warn and degrade | unchanged |

Level 4 is the only one reachable by a deployment whose references are
**generated at runtime**: there is no authored reference to annotate, and
every other level lives in code the operator does not deploy.

Level 2 sits above the code levels rather than below them because it is a
claim about one reference in particular, while `strict_resources` speaks for
references that said nothing. Only the same author's `$required: false`
overrides it.

Both `$required` and the `strict_resources` setting accept a boolean, or the
strings `"true"` / `"false"` in any case and with surrounding whitespace
ignored, so they work through `${VAR}` expansion. Any other value raises
rather than reading as lenient — a flag that silently means "off" is the
defect this vocabulary exists to close.

The environment setting is checked **when the environment is constructed**,
not when a resource turns out to be missing: `EnvironmentConfig(...)`,
`.load()` and `.from_dict()` all reject `strict_resources: "yes"` on the
spot. A malformed flag is malformed in every environment, and deferring the
check would surface it first in whichever deployment happened to lack a
resource — as an error about a *setting*. A value still spelled as a template
(`${STRICT}`, under `substitute_vars=False`) is left alone; it is not a value
yet.

(Unquoted `yes` in a YAML file is a *boolean* to the YAML parser, and arrives
here as `True` — accepted, and meaning strict. The rejected spelling is the
quoted string, which is also what `${VAR}` expansion produces.)

#### Marker keys are a closed set

The markers are `$resource`, `type`, `$requires` and `$required`. Everything
else in a reference block is an inline default, so a `$`-prefixed key outside
that set is rejected as a malformed reference rather than merged. Without
that guard `$requred: true` would silently mean *not required* — a typo one
character from the marker, reintroducing the silent degrade at the exact site
meant to close it.

Two corollaries, because a closed set is only as good as where it is checked:

- **`$requires` must be a list of names.** Its sibling `$required` takes a
  scalar, which makes `$requires: persistence` the natural slip. A bare
  string iterates character by character, so unvalidated it produced a check
  against letters — `missing required capabilities: ['c','e','i','n',...]`.
- **`$required` or `$requires` on a block with no `$resource` is rejected.**
  The guard above fires on a block that *is* a reference, and what makes one
  is the `$resource` key — so a typo in that key (`$resorce: conversations`)
  produced an ordinary dict that resolved to itself and reached the factory
  with its markers attached. A leftover policy marker is what gives it away.

`RESOURCE_MARKER_KEYS` is exported from `dataknobs_config` so a second reader
of this format has the vocabulary without copying the literal — and
`resolve_resource_references(config, environment, ...)` is exported so that a
consumer with a config tree and an environment does not have to *be* a second
reader in the first place.

#### Which exception

Every message names the dotted config path of the reference that failed, so
three references to `default` stay distinguishable in a log.

| Condition | Exception |
|---|---|
| Resource missing, policy strict | `ResourceNotFoundError` |
| Reference malformed — unknown `$` marker, unparseable `$required`, `$requires` that is not a list of names, or a policy marker with no `$resource` | `ConfigError` |
| Resource found but under-capable for `$requires` | `ConfigError` |
| A resource reaches itself (`a` → `b` → `a`) | `ConfigError` naming the cycle |

`ConfigBindingResolver.resolve()` raises `ResourceNotFoundError` for a
missing resource too. That API takes a `(type, name)` pair with no reference
to read a policy off, so it *is* the strict policy. Below the entry point the
two are the same code — `ConfigBindingResolver` resolves the config it looks
up with `resolve_resource_references`, so a reference *nested* inside a
resource gets the same marker validation, the same precedence chain and the
same cycle guard there as anywhere else, and can still declare
`$required: false` for itself. The two differ only at the top, where one has
a reference to read and the other does not.

!!! warning "`ResourceNotFoundError` is also a `KeyError`"

    It subclasses both `EnvironmentConfigError` and `KeyError`.
    `resolve_for_build()` could not previously raise a `KeyError`; under a
    strict policy it can. Code that wraps resolution in `except KeyError` for
    unrelated reasons will swallow it.

    `str(e)` returns the message as written. `KeyError.__str__` would
    otherwise return `repr(args[0])`, wrapping the sentence in quotes and
    escaping every name inside it; the type keeps both bases and overrides
    the rendering.

#### Preflight

`resolve_for_build(strict_resources=True)` resolves without constructing
anything, so it is safe to run at boot purely to prove every binding a config
names exists in this environment. It raises on the first failure.

Passing `strict_resources` with `resolve_resources=False` raises `ValueError`
rather than returning: the policy is read where references are resolved, so
that pair would check nothing and still hand back a config. (The *instance*
policy is a standing default rather than an assertion about one call, so
`EnvironmentAwareConfig(..., strict_resources=True)` is unaffected.)

To get *every* unresolvable reference in one pass — which is what an operator
auditing a config tree wants — use `find_unresolved_resources()`:

```yaml
# config/apps/my-bot.yaml — the reference declares its own policy
bot:
  knowledge_base:
    vector_store:
      $resource: knowledge
      type: vector_stores
      $required: true
```

```python
config = EnvironmentAwareConfig.load_app("my-bot")

for ref in config.find_unresolved_resources():
    print(f"{ref.path}: {ref.resource_type}/{ref.resource_name} "
          f"(required={ref.required}, defaults={ref.has_inline_defaults})")
# bot.knowledge_base.vector_store: vector_stores/knowledge (required=True, defaults=False)
```

It constructs nothing and raises nothing for a missing resource. A reference
that selects its resource by variable (`$resource: ${LLM_BINDING}`) is
reported under the **resolved** name, at any depth. Each entry is an
`UnresolvedResourceRef` with `path`, `resource_type`, `resource_name`,
`required`, and `has_inline_defaults` — the last distinguishing the two
degradations, since falling back to declared defaults is a config that still
builds while falling back to nothing is a factory about to be called with no
arguments. A reference at the root of the surveyed tree has `path == ""`, a
dotted path of zero segments.

It runs the **same walk** as `resolve_for_build`, differing only in what it
does when a resource is absent: record it and carry on down the lenient path,
rather than raise or warn. That is what makes it a prediction of the build
rather than a second opinion about it — a reference nested inside a resolved
resource is surveyed because a build reaches it, and one nested inside an
inline default the environment overrides is not, because a build discards it.

**An empty list means a build reaches no unresolvable reference.** Every
other way a reference can fail raises here instead of being listed: a
malformed reference, a resource that reaches itself, or a present resource
that does not declare a capability its reference `$requires`. Listing is for
the failure an operator fixes by adding bindings; for the rest there is no
complete list to give, because a config a build cannot walk cannot be walked
to the end here either. A survey that reported a tree sound while the build
raises on it would be worse than no survey.

### 5. Environment Detection

The system automatically detects the current environment via:

1. **Explicit**: `DATAKNOBS_ENVIRONMENT=production`
2. **Cloud indicators**: AWS Lambda, ECS, Kubernetes, GCP Cloud Run, Azure Functions
3. **Default**: `development`

## Classes

### EnvironmentConfig

Manages environment-specific resource bindings.

!!! note "Environment variable substitution"

    `EnvironmentConfig.load()` and `EnvironmentConfig.from_dict()` apply
    `${VAR}` / `${VAR:default}` substitution to every value by default,
    matching the behaviour of `InheritableConfigLoader.load()`. Required
    `${VAR}` refs without a default raise `ValueError` at load time. To
    preserve raw `${VAR}` literals (e.g., for inspection or
    transformation), pass the keyword-only `substitute_vars=False`.

```python
from dataknobs_config import EnvironmentConfig

# Auto-detect environment
env = EnvironmentConfig.load()

# Or specify explicitly
env = EnvironmentConfig.load("production", config_dir="config/environments")

# Preserve raw ${VAR} refs (e.g., for documentation or transformation)
env = EnvironmentConfig.load("production", substitute_vars=False)

# Get concrete config for a logical resource
db_config = env.get_resource("databases", "conversations")
# Returns: {"backend": "postgres", "connection_string": "..."}

# Check if resource exists
if env.has_resource("databases", "analytics"):
    analytics = env.get_resource("databases", "analytics")

# Get environment settings
log_level = env.get_setting("log_level", "INFO")
```

#### Environment File Format

```yaml
# config/environments/production.yaml
name: production
description: AWS production environment

settings:
  log_level: INFO
  enable_metrics: true
  strict_resources: true   # a reference to a resource this file does not
                           # define raises instead of degrading

resources:
  databases:
    default:
      backend: postgres
      connection_string: ${DATABASE_URL}
      pool_size: 20

    conversations:
      backend: postgres
      connection_string: ${DATABASE_URL}
      table: conversations

  vector_stores:
    default:
      backend: pgvector
      connection_string: ${DATABASE_URL}

    knowledge:
      backend: pgvector
      connection_string: ${DATABASE_URL}
      dimensions: 1536

  llm_providers:
    default:
      provider: openai
      model: gpt-4
      api_key: ${OPENAI_API_KEY}
      capabilities: [chat, function_calling, streaming]
```

### EnvironmentAwareConfig

Configuration with environment-aware resource resolution.

```python
from dataknobs_config import EnvironmentAwareConfig

# Load app config with auto-detected environment
config = EnvironmentAwareConfig.load_app(
    "my-bot",
    app_dir="config/apps",
    env_dir="config/environments"
)

# Get resolved config for object building (late binding happens here)
resolved = config.resolve_for_build()

# Resolve specific section
bot_config = config.resolve_for_build("bot")

# Get portable config for storage (no env vars resolved)
portable = config.get_portable_config()
```

#### Resolution order

`resolve_for_build` substitutes the **app config first**, then splices in
resource references:

1. `${VAR}` refs authored in the app config are expanded (late binding) —
   except a `$resource` reference's inline defaults, which are held back for
   step 2.
2. `$resource` references are resolved against the environment, whose own
   values were already expanded when it loaded. Each surviving inline default
   is expanded here, as it is spliced.

The order matters. Once resource values are spliced in, they are
indistinguishable from app-authored ones, and a substitution pass over the
merged result would expand the environment's values a **second** time —
re-reading the content of a value as a template. See
[Substitution runs once per source](environment-variables.md#substitution-runs-once-per-source).

An environment built directly (`EnvironmentConfig(name=..., resources=...)`)
or loaded with `substitute_vars=False` has not been expanded, so
`resolve_for_build` expands it — once. It does so **per resource, as each one
is spliced in**, not over the environment as a whole: a resource is still
separable at the splice point, which is the latest point it can be expanded.
Expanding the whole environment up front would read values no reference names,
so an unset required `${VAR}` in an unrelated resource would abort a build that
never looked at it. Your own `EnvironmentConfig` is never mutated by this.

A reference's inline defaults follow the same rule one level in. Step 1 holds
them back, and each is expanded at the splice — once, and only where the
environment did not supply the key. The splice is the latest point they are
still separable, so expanding one earlier would read a value the build then
discards: a dev-time fallback that production overrides would still have to
resolve in production, and an unset required `${VAR}` among them would abort a
build that never used it.

This covers a default's **value**, not the key that names it. What decides
whether a default is discarded is its key, so a key must be expanded to ask
the question — and deferring it would expand every default's key at the splice
instead, to ask it there. A required `${VAR}` in key position therefore does
have to resolve, even where the environment supplies that key.

That deferral is also what keeps a **nested** reference at one expansion —
though not always at the same step. Arriving inside an inline default, or
inside a resource an *unsubstituted* environment supplies, its own defaults
reach their own splice raw and are expanded there. Arriving inside a resource
an already-substituted environment supplies, they were expanded at that
environment's load, and `substituted` is what stops the splice expanding them
a second time.

Because step 1 runs first, the `$resource` and `type` values are themselves
substituted, so resource *selection* can be bound to an environment variable:

```yaml
llm:
  $resource: ${LLM_BINDING}    # expands, then resolves
  type: llm_providers
```

Before this ordering, the literal text `${LLM_BINDING}` was looked up as a
resource name, matched nothing, and silently fell back to the reference's
inline defaults.

#### Application Config Format

```yaml
# config/apps/my-bot.yaml
name: my-bot
version: "1.0.0"

bot:
  llm:
    $resource: default
    type: llm_providers
    $requires: [function_calling]  # Optional: require specific capabilities
    temperature: 0.7               # Merged into resolved config

  conversation_storage:
    $resource: conversations
    type: databases

  knowledge_base:
    vector_store:
      $resource: knowledge
      type: vector_stores

  system_prompt: |
    You are a helpful assistant.
```

### ConfigBindingResolver

Resolves logical resource bindings to concrete instances using factories.

```python
from dataknobs_config import (
    EnvironmentConfig,
    ConfigBindingResolver,
    SimpleFactory,
    CallableFactory,
    AsyncCallableFactory,
)

# Load environment
env = EnvironmentConfig.load("production")

# Create resolver
resolver = ConfigBindingResolver(env)

# Register factories for resource types
resolver.register_factory("databases", SimpleFactory(DatabaseConnection))
resolver.register_factory("vector_stores", CallableFactory(create_vector_store))

# Resolve a logical reference to a concrete instance
db = resolver.resolve("databases", "conversations")

# With config overrides
db = resolver.resolve("databases", "conversations", pool_size=50)

# Async resolution (for async factories)
vector_store = await resolver.resolve_async("vector_stores", "knowledge")

# Skip cache for fresh instance
fresh_db = resolver.resolve("databases", "conversations", use_cache=False)

# Clear cache
resolver.clear_cache()  # All resources
resolver.clear_cache("databases")  # Specific type
```

#### Factory Types

**SimpleFactory** - Creates instances of a class:

```python
from dataknobs_config import SimpleFactory

resolver.register_factory(
    "databases",
    SimpleFactory(DatabaseConnection, timeout=30)  # Default kwargs
)
```

**CallableFactory** - Wraps a callable:

```python
from dataknobs_config import CallableFactory

def create_database(backend, connection_string, **kwargs):
    if backend == "postgres":
        return PostgresDB(connection_string, **kwargs)
    elif backend == "sqlite":
        return SQLiteDB(connection_string, **kwargs)

resolver.register_factory("databases", CallableFactory(create_database))
```

**AsyncCallableFactory** - Wraps an async callable:

```python
from dataknobs_config import AsyncCallableFactory

async def create_database(backend, connection_string, **kwargs):
    db = DatabaseConnection(backend, connection_string)
    await db.connect()
    return db

resolver.register_factory("databases", AsyncCallableFactory(create_database))

# Must use resolve_async
db = await resolver.resolve_async("databases", "conversations")
```

## Usage Patterns

### Pattern 1: Direct Environment Config

For simple resource lookup without object instantiation:

```python
from dataknobs_config import EnvironmentConfig

env = EnvironmentConfig.load()

# Get config for a resource
db_config = env.get_resource("databases", "conversations")

# Use config directly
connection = create_connection(**db_config)
```

### Pattern 2: App Config with Late Binding

For applications with portable configuration:

```python
from dataknobs_config import EnvironmentAwareConfig

# Load portable app config
config = EnvironmentAwareConfig.load_app("my-app")

# Store portable config (safe for database storage)
db.store(config.get_portable_config())

# At runtime, resolve for current environment
resolved = config.resolve_for_build()
app = MyApp.from_config(resolved)
```

### Pattern 3: Full Factory Resolution

For applications needing complete object lifecycle management:

```python
from dataknobs_config import (
    EnvironmentConfig,
    ConfigBindingResolver,
    AsyncCallableFactory,
)

async def create_bot_resources():
    env = EnvironmentConfig.load()
    resolver = ConfigBindingResolver(env)

    # Register factories
    resolver.register_factory("databases", AsyncCallableFactory(create_db))
    resolver.register_factory("llm_providers", AsyncCallableFactory(create_llm))

    # Resolve resources
    db = await resolver.resolve_async("databases", "conversations")
    llm = await resolver.resolve_async("llm_providers", "default")

    return {"db": db, "llm": llm}
```

## Environment Detection

The environment is automatically detected in this order:

| Priority | Method | Example |
|----------|--------|---------|
| 1 | `DATAKNOBS_ENVIRONMENT` | `export DATAKNOBS_ENVIRONMENT=production` |
| 2 | AWS Lambda/ECS | `AWS_EXECUTION_ENV` present |
| 3 | AWS ECS Fargate | `ECS_CONTAINER_METADATA_URI` present |
| 4 | Kubernetes | `KUBERNETES_SERVICE_HOST` present |
| 5 | Google Cloud Run | `K_SERVICE` present |
| 6 | Azure Functions | `FUNCTIONS_WORKER_RUNTIME` present |
| 7 | Default | `development` |

For cloud environments, the actual environment name (staging, production) is read from the `ENVIRONMENT` variable if present.

The detected name becomes a file name under the environment config
directory, so it is bounded by that directory: a name may address a
subdirectory (`tier/production`), but one that *lands* outside — whether
spelled with `..` or as an absolute path — raises
`EnvironmentConfigError` rather than reading the file it points at. The
same bound applies to the `app_name` passed to
`EnvironmentAwareConfig.load_app`, which raises
`EnvironmentAwareConfigError`. A missing environment file is unaffected:
`EnvironmentConfig.load` still returns an empty configuration for an
environment that has no file.

Both entry points take `allow_outside=True` to lift the bound for a
layout that genuinely spans sibling trees:

```python
EnvironmentConfig.load(env, config_dir, allow_outside=True)
EnvironmentAwareConfig.load_app(app, app_dir, env_dir, allow_outside=True)
```

Weigh it here more carefully than elsewhere. Called without an explicit
`environment`, these read the name from `DATAKNOBS_ENVIRONMENT` or
`ENVIRONMENT`, so opting out widens what a process environment variable
can address; and `load_app` applies the flag to **both** of its lookups,
the app name and the environment name, rather than silently covering
only one. Off by default, and an escaping name is logged at WARNING.

## Best Practices

### 1. Store Portable Configs

Only store configs with `$resource` references in databases and registries:

```python
# CORRECT: Store portable config
registry.store(domain_id, config.get_portable_config())

# At load time, resolve for current environment
stored = registry.load(domain_id)
config = EnvironmentAwareConfig(stored)
resolved = config.resolve_for_build()
```

### 2. Use Late Binding

Resolve environment variables at instantiation time, not load time:

```python
# Loading keeps placeholders
config = EnvironmentAwareConfig.load_app("my-app")
# config still has ${DATABASE_URL} placeholders

# Resolution substitutes env vars
resolved = config.resolve_for_build()
# resolved has actual connection strings
```

### 3. Consistent Resource Names

Use the same logical names across all environment configs:

```yaml
# development.yaml
resources:
  databases:
    conversations: ...  # Same name

# production.yaml
resources:
  databases:
    conversations: ...  # Same name
```

### 4. Separate Behavior from Infrastructure

Put behavioral settings and capability requirements in app configs, infrastructure and capability metadata in environment configs:

```yaml
# App config (portable)
llm:
  $resource: default
  type: llm_providers
  $requires: [function_calling]  # What the app needs
  temperature: 0.7               # Behavioral
  max_tokens: 2000               # Behavioral

# Environment config (per-environment)
llm_providers:
  default:
    provider: openai              # Infrastructure
    api_key: ${KEY}               # Infrastructure
    capabilities: [chat, function_calling, streaming]  # What it provides
```

## API Reference

### EnvironmentConfig

| Method | Description |
|--------|-------------|
| `load(environment, config_dir, *, substitute_vars=True)` | Load environment config from file |
| `from_dict(data, *, substitute_vars=True)` | Create from dictionary |
| `substituted` (attribute) | Whether `${VAR}` refs in these values have been expanded |
| `substituted_view()` | An expanded copy of an unexpanded config |
| `detect_environment()` | Detect current environment |
| `get_resource(type, name, defaults, *, required=None)` | Get resource config. `required` decides the absent case independently of `defaults` |
| `has_resource(type, name)` | Check if resource exists |
| `get_setting(key, default)` | Get environment setting |
| `get_resource_types()` | List all resource types |
| `get_resource_names(type)` | List resources of a type |
| `merge(other)` | Merge with another config |
| `to_dict()` | Export as dictionary |

### EnvironmentAwareConfig

| Method | Description |
|--------|-------------|
| `load_app(name, app_dir, env_dir, environment, *, allow_outside=False, strict_resources=None)` | Load app with environment |
| `from_dict(config, environment, env_dir, *, strict_resources=None)` | Create from dictionary |
| `resolve_for_build(key, resolve_resources, resolve_env_vars, *, strict_resources=None)` | Late-bind config |
| `find_unresolved_resources(config_key, *, strict_resources=None)` | Survey every unresolvable reference; builds nothing |
| `strict_resources` (property) | The instance-level missing-resource policy, or `None` to defer |
| `get_portable_config()` | Get unresolved config |
| `get(key, default)` | Get config value |
| `with_environment(environment, env_dir)` | Create with different env; carries `strict_resources` |
| `get_resource(type, name, defaults, *, required=None)` | Direct resource access |
| `get_setting(key, default)` | Direct setting access |

### ConfigBindingResolver

| Method | Description |
|--------|-------------|
| `register_factory(type, factory)` | Register resource factory |
| `unregister_factory(type)` | Remove factory |
| `has_factory(type)` | Check if factory registered |
| `get_registered_types()` | List registered types |
| `resolve(type, name, use_cache, **overrides)` | Resolve to instance |
| `resolve_async(type, name, use_cache, **overrides)` | Async resolution |
| `get_cached(type, name)` | Get cached instance |
| `is_cached(type, name)` | Check if cached |
| `clear_cache(type)` | Clear cache |
| `cache_instance(type, name, instance)` | Manually cache |

### Module-level

| Function | Description |
|----------|-------------|
| `resolve_resource_references(config, environment, *, substitute=False, strict_resources=None)` | Resolve every `$resource` reference in a config tree |
| `RESOURCE_MARKER_KEYS` | The closed marker set — `$resource`, `type`, `$requires`, `$required` |
| `STRICT_RESOURCES_SETTING` | The environment settings key holding the policy |
| `UnresolvedResourceRef` | A survey finding: `path`, `resource_type`, `resource_name`, `required`, `has_inline_defaults` |

`resolve_resource_references` is the shared primitive behind both resolvers.
Reach for it rather than walking a config for `$resource` blocks yourself: a
hand-written reader is how one arrived that recognised only `$resource` and
`type`, so it discarded every inline default, ignored `$required`, let a
misspelled marker through as data, and left a reference nested inside a
resolved resource as a literal dict for whatever read the config next.

## See Also

- [Configuration System](configuration-system.md) - Core Config class
- [Environment Variables](environment-variables.md) - Environment override system
- [Factory Registration](factory-registration.md) - Object construction patterns

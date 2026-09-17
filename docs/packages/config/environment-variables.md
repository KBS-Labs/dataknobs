# Environment Variables

The DataKnobs Config package provides comprehensive environment variable support for overriding configuration values at runtime. This enables secure management of sensitive data and environment-specific configuration without modifying files.

## Overview

Environment variables can override any configuration value using a structured naming convention. The system supports:

- Automatic type conversion
- Both named and indexed access
- Variable substitution in configuration files

Overrides address a **top-level attribute** of a configuration item. Reaching
into a nested mapping is not supported -- see [Nesting](#nesting) below for
what happens if you try.

## Naming Convention

Environment variables follow this pattern:

```
DATAKNOBS_<TYPE>__<NAME_OR_INDEX>__<ATTRIBUTE>
```

- **DATAKNOBS**: The prefix. Fixed for `Config`; see
  [Prefix and Selection](#prefix-and-selection)
- **TYPE**: The configuration type, **spelled exactly as the config file's
  top-level key is**, upper-cased. A file declaring `databases:` is addressed
  by `DATAKNOBS_DATABASES__`, not `DATAKNOBS_DATABASE__` -- the name is matched
  verbatim, with no singular/plural reconciliation, and a variable naming a
  type that does not exist is logged at WARNING and skipped
- **NAME_OR_INDEX**: The item's `name`, or a numeric index (negative allowed)
- **ATTRIBUTE**: One top-level attribute of that item. Not a path -- see
  [Nesting](#nesting)

### Examples

These address a file declaring `databases:` and `caches:`:

```bash
# Override database host by name
DATAKNOBS_DATABASES__PRIMARY__HOST=prod.example.com

# Override database port by index
DATAKNOBS_DATABASES__0__PORT=5433

# Override cache TTL
DATAKNOBS_CACHES__REDIS__TTL=7200
```

### Nesting

Everything after the second separator is folded back into a single attribute
name, and that name is then walked one segment at a time. So
`DATAKNOBS_DATABASES__PRIMARY__CONNECTION__TIMEOUT` reaches
`connection.timeout`:

```python
# config.yaml has databases[primary].connection.timeout: 30
os.environ["DATAKNOBS_DATABASES__PRIMARY__CONNECTION__TIMEOUT"] = "60"

config = Config.from_file("config.yaml")
print(config.get("databases", "primary"))
# {'name': 'primary', 'connection': {'timeout': 60}, 'type': 'databases'}
```

Every segment but the last has to name something that is already there -- a
key of a mapping, or an in-range index of a list. The last segment is written,
and a mapping gains it if it is absent, which is the rule a single-segment
attribute has always followed.

A path that does not resolve is logged and dropped. Nothing is written
anywhere, which is the part worth relying on: an override that misses leaves
no trace in the configuration rather than a key beside the value you meant to
change.

```python
os.environ["DATAKNOBS_DATABASES__PRIMARY__NOPE__TIMEOUT"] = "60"

config = Config.from_file("config.yaml")
# WARNING  Failed to apply environment override
#          xref:databases[primary].nope__timeout: 'nope__timeout' does not
#          resolve in databases[primary], so nothing was written

print(config.get("databases", "primary"))
# {'name': 'primary', 'connection': {'timeout': 30}, 'type': 'databases'}
```

## Type Conversion

Values are automatically converted to appropriate types:

```bash
# String (default)
DATAKNOBS_DATABASES__PRIMARY__HOST=localhost

# Integer
DATAKNOBS_DATABASES__PRIMARY__PORT=5432

# Float
DATAKNOBS_SERVICES__API__TIMEOUT=30.5

# Boolean (true, false, yes, no, 1, 0)
DATAKNOBS_DATABASES__PRIMARY__SSL_ENABLED=true
DATAKNOBS_SERVICES__API__DEBUG=1
```

## Applying Environment Overrides

### During Configuration Load

Overrides are applied **at construction, automatically**. There is no
`apply_env_overrides()` to call afterwards and no flag to switch it on:

```python
from dataknobs_config import Config

# Environment overrides are already applied
config = Config.from_file("config.yaml")
```

### Opting Out

Pass `use_env=False`. The constructor and both classmethods take it:

```python
from dataknobs_config import Config

config = Config("config.yaml", use_env=False)
config = Config.from_file("config.yaml", use_env=False)
config = Config.from_dict(declared, use_env=False)
```

It is a declared keyword, so a misspelling raises `TypeError` rather than
leaving the overrides silently on.

### Prefix and Selection

The prefix is fixed at `DATAKNOBS_`, and every matching variable is applied.
Selecting a subset, filtering by name, or using a custom prefix are not
exposed through `Config`; a variable is either named so that it addresses an
existing item or it is silently skipped.

To narrow what can be applied, control the environment rather than the load --
unset the variables you do not want, or run with a curated environment.

## Variable Substitution in Files

Configuration files can reference environment variables directly. The
canonical helper is :func:`substitute_env_vars` from
``dataknobs_config.inheritance`` (re-exported as
``dataknobs_config.substitute_env_vars``); it powers
``InheritableConfigLoader.load``, ``EnvironmentConfig.load`` /
``from_dict``, ``EnvironmentAwareConfig.resolve_for_build``,
``ConfigBindingResolver._get_resolved_config``, and ``Config._load_dict``.

### Supported syntax

| Syntax | Behavior |
|---|---|
| ``${VAR}`` | Required. Raises ``RequiredEnvVarError`` if ``VAR`` is unset. |
| ``${VAR:default}`` | Uses ``default`` when ``VAR`` is unset (DataKnobs legacy form). |
| ``${VAR:-default}`` | Bash-style alias for ``${VAR:default}``. |
| ``${VAR:?error_msg}`` | Bash-style. When ``VAR`` is unset, raises ``RequiredEnvVarError("Required environment variable not set: <error_msg>")`` (the variable name is used in place of ``<error_msg>`` when ``error_msg`` is empty). |

``RequiredEnvVarError`` is a subclass of ``ValueError``, so existing
``except ValueError`` / ``pytest.raises(ValueError)`` continue to catch
required-but-unset failures. Catch ``RequiredEnvVarError`` directly to
inspect ``var_name``, ``bash_form`` (``True`` for the ``${VAR:?msg}``
form, ``False`` for the bare ``${VAR}`` form), and ``explicit_message``.

### Basic Substitution

```yaml
database:
  host: ${DB_HOST}
  password: ${DB_PASSWORD}
```

### With Default Values

```yaml
database:
  # Colon syntax (DataKnobs legacy)
  host: ${DB_HOST:localhost}
  port: ${DB_PORT:5432}

  # Bash-style with default
  user: ${DB_USER:-postgres}
  password: ${DB_PASS:-}

  # Bash-style required-with-message — fails fast at load time
  api_key: ${API_KEY:?API_KEY must be set for production}
```

### Nested Substitution

```yaml
database:
  connection_string: ${DB_PROTOCOL:postgresql}://${DB_HOST}:${DB_PORT}/${DB_NAME}
```

### Helper options

``substitute_env_vars`` exposes three keyword-only options:

| Option | Default | Effect |
|---|---|---|
| ``type_coerce`` | ``False`` | When a string is *entirely* a single ``${VAR}`` placeholder, coerce the resolved value to ``int`` / ``float`` / ``bool`` if the literal looks like one. Mixed-content strings (``"port=${PORT}"``) always return strings. ``Config._load_dict`` passes ``type_coerce=True`` to preserve historical behavior; other loaders default to ``False`` (string out). |
| ``expand_user_paths`` | ``True`` | Apply ``os.path.expanduser`` to substituted strings so ``${PATH_VAR}`` whose value is ``~/foo`` yields ``/home/.../foo``. ``os.path.expanduser`` is a no-op on strings that do not start with ``~``, so URLs and connection strings (``postgresql://host:5432/db``) pass through unchanged. Set to ``False`` for strict no-touch substitution. |
| ``substitute_keys`` | ``True`` | Substitute ``${VAR}`` references in dict keys as well as values. Keys are never type-coerced even when ``type_coerce=True``. |

```python
from dataknobs_config import substitute_env_vars

# Strict no-touch substitution
result = substitute_env_vars(data, expand_user_paths=False)

# Coerce numeric env vars to int/float, leave keys as literal strings
result = substitute_env_vars(data, type_coerce=True, substitute_keys=False)
```

### Migrating from ``VariableSubstitution``

The class :class:`VariableSubstitution` is a deprecated thin shim over
``substitute_env_vars``. New code should call ``substitute_env_vars``
directly:

| Old | New |
|---|---|
| ``VariableSubstitution().substitute(data)`` | ``substitute_env_vars(data, type_coerce=True, expand_user_paths=False, substitute_keys=False)`` |

Constructing ``VariableSubstitution()`` emits a ``DeprecationWarning``;
the class will be removed in a future release.

--8<-- "packages/config/docs/environment-variable-substitution.md:substitute-once"

## Named vs Indexed Access

### Named Access

Use the configuration item's name:

```bash
# config.yaml:
# databases:
#   - name: primary
#     host: localhost

DATAKNOBS_DATABASES__PRIMARY__HOST=prod.example.com
```

### Indexed Access

Use numeric indices (0-based):

```bash
# First database
DATAKNOBS_DATABASES__0__HOST=prod.example.com

# Second database
DATAKNOBS_DATABASES__1__HOST=analytics.example.com

# Last database (negative indexing)
DATAKNOBS_DATABASES__-1__HOST=backup.example.com
```

## Nested Attributes and List Elements

Both are addressable, and by the same walk -- this is the
[Nesting](#nesting) rule seen from the two directions people most often try.

A nested mapping:

```bash
# config.yaml:
# databases:
#   - name: search
#     settings:
#       number_of_shards: 3

# settings.number_of_shards becomes 5.
DATAKNOBS_DATABASES__SEARCH__SETTINGS__NUMBER_OF_SHARDS=5
```

A list element, addressed by index:

```bash
# config.yaml:
# services:
#   - name: api
#     allowed_origins:
#       - http://localhost:3000

# allowed_origins becomes ['https://app.example.com'].
DATAKNOBS_SERVICES__API__ALLOWED_ORIGINS__0=https://app.example.com
```

A list is never extended to make an override fit. An absent mapping key is
created, because there is somewhere obvious to put it; an absent list position
is not, because appending would put the value somewhere you did not name. So
`ALLOWED_ORIGINS__7` against a one-element list is logged and dropped, and the
list is the one from the file.

Substituting the value in the file itself with
[`${VAR}`](#variable-substitution-in-files) remains the other way to reach a
nested value, and is still the only one that works on a document whose shape
is not known in advance -- it is applied to the file's own text before the
config is built.

## Complex Examples

### Database Configuration

```bash
# Development
export DATAKNOBS_DATABASES__PRIMARY__HOST=localhost
export DATAKNOBS_DATABASES__PRIMARY__PORT=5432
export DATAKNOBS_DATABASES__PRIMARY__USERNAME=dev_user
export DATAKNOBS_DATABASES__PRIMARY__PASSWORD=dev_pass

# Production
export DATAKNOBS_DATABASES__PRIMARY__HOST=prod-db.example.com
export DATAKNOBS_DATABASES__PRIMARY__PORT=5432
export DATAKNOBS_DATABASES__PRIMARY__USERNAME=prod_user
export DATAKNOBS_DATABASES__PRIMARY__PASSWORD=${SECRET_DB_PASSWORD}
export DATAKNOBS_DATABASES__PRIMARY__SSL_ENABLED=true
export DATAKNOBS_DATABASES__PRIMARY__POOL_SIZE=50
```

### Service Configuration

```bash
# API Service
export DATAKNOBS_SERVICES__API__PORT=8000
export DATAKNOBS_SERVICES__API__HOST=0.0.0.0
export DATAKNOBS_SERVICES__API__DEBUG=false
export DATAKNOBS_SERVICES__API__LOG_LEVEL=INFO
export DATAKNOBS_SERVICES__API__RATE_LIMIT=1000

# Worker Service
export DATAKNOBS_SERVICES__WORKER__CONCURRENCY=10
export DATAKNOBS_SERVICES__WORKER__QUEUE_NAME=tasks
export DATAKNOBS_SERVICES__WORKER__RETRY_ATTEMPTS=3
```

### Cache Configuration

```bash
# Redis Cache
export DATAKNOBS_CACHES__REDIS__HOST=redis.example.com
export DATAKNOBS_CACHES__REDIS__PORT=6379
export DATAKNOBS_CACHES__REDIS__DB=0
export DATAKNOBS_CACHES__REDIS__TTL=3600
export DATAKNOBS_CACHES__REDIS__MAX_CONNECTIONS=100
```

## Docker and Container Usage

### Docker Compose

```yaml
version: '3.8'
services:
  app:
    image: myapp:latest
    environment:
      - DATAKNOBS_DATABASES__PRIMARY__HOST=db
      - DATAKNOBS_DATABASES__PRIMARY__PORT=5432
      - DATAKNOBS_DATABASES__PRIMARY__USERNAME=postgres
      - DATAKNOBS_DATABASES__PRIMARY__PASSWORD=${DB_PASSWORD}
      - DATAKNOBS_CACHES__REDIS__HOST=redis
      - DATAKNOBS_SERVICES__API__PORT=8000
```

### Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  DATAKNOBS_DATABASES__PRIMARY__HOST: "postgres-service"
  DATAKNOBS_DATABASES__PRIMARY__PORT: "5432"
  DATAKNOBS_CACHES__REDIS__HOST: "redis-service"
  DATAKNOBS_SERVICES__API__LOG_LEVEL: "INFO"
```

### Kubernetes Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
stringData:
  DATAKNOBS_DATABASES__PRIMARY__PASSWORD: "secret-password"
  DATAKNOBS_SERVICES__API__SECRET_KEY: "secret-api-key"
```

## .env File Support

Use .env files for local development:

```bash
# .env
DATAKNOBS_DATABASES__PRIMARY__HOST=localhost
DATAKNOBS_DATABASES__PRIMARY__PORT=5432
DATAKNOBS_DATABASES__PRIMARY__USERNAME=dev_user
DATAKNOBS_DATABASES__PRIMARY__PASSWORD=dev_password
DATAKNOBS_CACHES__REDIS__HOST=localhost
DATAKNOBS_CACHES__REDIS__PORT=6379
DATAKNOBS_SERVICES__API__DEBUG=true
DATAKNOBS_SERVICES__API__LOG_LEVEL=DEBUG
```

Load with python-dotenv:

```python
from dotenv import load_dotenv
from dataknobs_config import Config

# Load .env file FIRST -- Config reads the environment as it is built, so a
# .env loaded afterwards arrives too late to override anything.
load_dotenv()

config = Config.from_file("config.yaml")
```

## Debugging Environment Variables

### List Applied Overrides

`Config` does not report what it applied. Ask the override reader the same
question it asks the environment, before building the config:

```python
from dataknobs_config.environment import EnvironmentOverrides

overrides = EnvironmentOverrides().get_overrides()

print("Overrides visible in the environment:")
for ref, value in overrides.items():
    print(f"  {ref}: {value!r}")
# xref:databases[primary].host: 'db.prod.internal'
# xref:databases[primary].pool_size: 20
```

That lists what the environment *offers*. An entry naming a type or item the
config does not contain is logged at WARNING and skipped, so turn logging on
to see the difference between offered and applied:

```python
import logging
logging.basicConfig(level=logging.WARNING)

config = Config.from_file("config.yaml")
# WARNING  Failed to apply environment override xref:caches[redis].ttl: ...
```

### Validate Environment Variables

```python
def validate_env_overrides(config):
    """Validate that required environment variables are set."""
    required = [
        "DATAKNOBS_DATABASES__PRIMARY__PASSWORD",
        "DATAKNOBS_SERVICES__API__SECRET_KEY",
    ]
    
    missing = []
    for var in required:
        if var not in os.environ:
            missing.append(var)
    
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")

# Use BEFORE building the config -- by the time you hold one, the overrides
# have already been applied.
validate_env_overrides(config)
config = Config.from_file("config.yaml")
```

## Best Practices

### 1. Security

- Never commit sensitive environment variables to version control
- Use secrets management systems in production
- Validate that required secrets are set before starting

### 2. Naming

- Use consistent, descriptive names
- Group related variables with common prefixes
- Document all environment variables

### 3. Defaults

- Provide sensible defaults in configuration files
- Use environment variables for overrides, not base configuration
- Document which values are commonly overridden

### 4. Type Safety

```python
# Validate types after applying overrides
def validate_types(config):
    db_config = config.get("databases", "primary")
    assert isinstance(db_config["port"], int)
    assert isinstance(db_config["ssl_enabled"], bool)
```

### 5. Documentation

Create an environment variable reference:

```markdown
# Environment Variables Reference

## Database Configuration
- `DATAKNOBS_DATABASES__PRIMARY__HOST`: Database host (default: localhost)
- `DATAKNOBS_DATABASES__PRIMARY__PORT`: Database port (default: 5432)
- `DATAKNOBS_DATABASES__PRIMARY__USER`: Database user (required)
- `DATAKNOBS_DATABASES__PRIMARY__PASSWORD`: Database password (required)

## Cache Configuration
- `DATAKNOBS_CACHES__REDIS__HOST`: Redis host (default: localhost)
- `DATAKNOBS_CACHES__REDIS__PORT`: Redis port (default: 6379)
```

## Troubleshooting

### Common Issues

1. **Variables Not Applied**: They are applied at construction -- check the
   variable was set *before* the `Config` was built, that `use_env=False` was
   not passed, and that the TYPE segment matches the config file's top-level
   key exactly (`databases:` is `DATAKNOBS_DATABASES__`, never
   `DATAKNOBS_DATABASE__`)
2. **Wrong Type**: Check automatic type conversion is working as expected.
   Note that `0` and `1` become booleans, not integers
3. **Name Mismatch**: Verify configuration item names match environment
   variable names. A variable naming an item that does not exist is skipped
   with a WARNING, not an error
4. **Nothing Beyond the Second Separator**: `A__B__C__D` sets an attribute
   literally named `c__d`; see [Nesting](#nesting)
5. **Case Sensitivity**: The variable name is upper-case by convention; the
   type and attribute are lower-cased when parsed, so the config keys they
   address are matched in lower case

### Debug Mode

There is no debug flag. Failures to apply an override are logged at WARNING by
`dataknobs_config.config`:

```python
import logging
logging.basicConfig(level=logging.WARNING)

config = Config.from_file("config.yaml")
```

## Advanced Usage

### Custom Override Logic

Overrides are applied inside `Config.__init__`, so there is no post-load hook
to override. Build with `use_env=False` and drive `EnvironmentOverrides`
yourself when you need to preprocess or filter:

```python
from dataknobs_config import Config
from dataknobs_config.environment import EnvironmentOverrides


def load_with_filtered_overrides(path, *, prefix="DATAKNOBS_", skip=()):
    """Apply every override except the ones named in `skip`."""
    config = Config(path, use_env=False)
    reader = EnvironmentOverrides(prefix=prefix)

    for ref, value in reader.get_overrides().items():
        type_name, name_or_index, attr = reader.parse_env_reference(ref)
        if attr in skip:
            continue
        item = config.get(type_name, name_or_index)
        item[attr] = value
        config.set(type_name, name_or_index, item)

    return config


# A custom prefix is reachable this way, and only this way.
config = load_with_filtered_overrides("config.yaml", prefix="MYAPP_", skip=("password",))
```

### Dynamic Environment Variables

```python
import os

def set_dynamic_env_vars(environment):
    """Set environment variables based on deployment environment."""
    if environment == "production":
        os.environ["DATAKNOBS_DATABASES__PRIMARY__POOL_SIZE"] = "50"
        os.environ["DATAKNOBS_SERVICES__API__WORKERS"] = "4"
    else:
        os.environ["DATAKNOBS_DATABASES__PRIMARY__POOL_SIZE"] = "10"
        os.environ["DATAKNOBS_SERVICES__API__WORKERS"] = "1"

# Again: set the variables first, then build.
set_dynamic_env_vars("production")
config = Config.from_file("config.yaml")
```
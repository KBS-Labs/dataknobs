# Environment Variable Substitution

The canonical helper for environment-variable substitution in
configuration data is `substitute_env_vars`, exported as
`dataknobs_config.substitute_env_vars` (canonical implementation lives
in `dataknobs_config.inheritance`).

It is invoked by every loader in the package:

| Loader | Substitution flags |
|---|---|
| `InheritableConfigLoader.load` | defaults (string out, tilde expansion on, keys substituted) |
| `EnvironmentConfig.load` / `from_dict` | defaults (controlled via `substitute_vars=True`) |
| `EnvironmentAwareConfig.resolve_for_build` | defaults |
| `ConfigBindingResolver._get_resolved_config` | defaults |
| `Config._load_dict` | `type_coerce=True, expand_user_paths=False, substitute_keys=False` |

## Syntax

The pattern is a bash superset:

| Syntax | Behavior |
|---|---|
| `${VAR}` | Required. Raises `RequiredEnvVarError` if `VAR` is unset. |
| `${VAR:default}` | Uses `default` when `VAR` is unset (DataKnobs legacy form). |
| `${VAR:-default}` | Bash-style alias for `${VAR:default}`. |
| `${VAR:?error_msg}` | Bash-style. When `VAR` is unset, raises `RequiredEnvVarError("Required environment variable not set: <error_msg>")` (the variable name is used in place of `<error_msg>` when `error_msg` is empty). |

Substitution applies to nested dicts and lists. Non-string dict keys
(integers, booleans) pass through unchanged.

`RequiredEnvVarError` is a subclass of `ValueError`, so existing
`except ValueError` / `pytest.raises(ValueError)` continue to catch
required-but-unset failures. Catch `RequiredEnvVarError` directly when
you need to inspect the failure: it carries `var_name` (the unset
variable), `bash_form` (`True` for the `${VAR:?msg}` form, `False` for
the bare `${VAR}` form), and `explicit_message` (the user-supplied
message from `${VAR:?msg}`, or `None`).

## Options

```python
from dataknobs_config import substitute_env_vars

substitute_env_vars(
    data,
    *,
    type_coerce: bool = False,
    expand_user_paths: bool = True,
    substitute_keys: bool = True,
)
```

| Option | Default | Effect |
|---|---|---|
| `type_coerce` | `False` | When an entire string is a single `${VAR}` placeholder, coerce the value to `int` / `float` / `bool` when it looks like one. Mixed-content strings (`"port=${PORT}"`) remain strings. |
| `expand_user_paths` | `True` | Apply `os.path.expanduser` to substituted strings. Leaves URLs and connection strings (`postgresql://host:5432/db`) intact because `os.path.expanduser` only touches strings that begin with `~`. Set to `False` for strict no-touch substitution. |
| `substitute_keys` | `True` | Substitute `${VAR}` in dict keys as well as values. Keys are never type-coerced even when `type_coerce=True`. |

## Migrating from `VariableSubstitution`

The class `VariableSubstitution` is a deprecated thin shim over
`substitute_env_vars`. It emits `DeprecationWarning` on construction
and will be removed in a future release. New code should call the
canonical helper directly:

```python
# Old
from dataknobs_config import VariableSubstitution
result = VariableSubstitution().substitute(data)

# New
from dataknobs_config import substitute_env_vars
result = substitute_env_vars(
    data,
    type_coerce=True,
    expand_user_paths=False,
    substitute_keys=False,
)
```

## Examples

```python
import os
from dataknobs_config import substitute_env_vars

os.environ["DB_HOST"] = "localhost"
os.environ["DB_PORT"] = "5432"

# Default flags: string out, tilde expansion on, keys substituted
substitute_env_vars({"dsn": "postgresql://${DB_HOST}:${DB_PORT}/db"})
# -> {"dsn": "postgresql://localhost:5432/db"}

# type_coerce=True turns whole-value placeholders into typed primitives
substitute_env_vars({"port": "${DB_PORT}"}, type_coerce=True)
# -> {"port": 5432}

# Bash-style required-with-message
substitute_env_vars({"key": "${API_KEY:?API_KEY must be set}"})
# raises ValueError: Required environment variable not set: API_KEY must be set
```

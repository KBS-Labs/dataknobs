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
| `EnvironmentAwareConfig.resolve_for_build` | defaults, applied to the **app config** and to an environment that was not substituted at load |
| `ConfigBindingResolver._get_resolved_config` | defaults, applied to **overrides** and to an environment that was not substituted at load |
| `Config._load_dict` | `type_coerce=True, expand_user_paths=False, substitute_keys=False` |

Several of these compose — an app config resolves against an environment,
and the environment was substituted when it loaded. See
[Substitution runs once per source](#substitution-runs-once-per-source) for
the rule that keeps a value from being expanded twice.

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

<!-- --8<-- [start:substitute-once] -->
## Substitution runs once per source

Substitution is **not idempotent**. Running it twice over the same data
expands the *output* of the first pass, so a value whose own text contains
`${...}` is re-read as a template and replaced by whatever unrelated
variable that text happens to name:

```python
os.environ["DB_PASSWORD"] = "p${x}ss"   # a perfectly ordinary password
os.environ["x"] = "INJECTED"

substitute_env_vars({"password": "${DB_PASSWORD}"})
# -> {"password": "p${x}ss"}          correct

substitute_env_vars(substitute_env_vars({"password": "${DB_PASSWORD}"}))
# -> {"password": "pINJECTEDss"}      the secret is now a different string
```

Generated passwords routinely contain `$` and `{`, so this is not exotic.
The rule the package follows is:

> **Substitute each source exactly once, at the latest point that source is
> still separable.**

For an environment that point is its load; for an app config it is entry to
`EnvironmentAwareConfig.resolve_for_build`, which is still before resource
references are spliced in. Once spliced, the two sources are merged beyond
telling apart, and any pass over the result would expand the environment's
values a second time.

The surfaces in this package that the rule governs, and where each records
what it has already done:

| Surface | Source | Provenance recorded in |
|---|---|---|
| `EnvironmentConfig` | an environment file | `substituted`, on the object |
| `EnvironmentAwareConfig.resolve_for_build` | the app config | the ordering — substituted on entry, before the resource splice |
| `ConfigBindingResolver` | a resolved resource, plus caller `**overrides` | the environment's `substituted`; overrides are a separate source with their own single pass |
| `InheritableConfigLoader` | a config file and the parents it `extends:` | the cache key |

### `EnvironmentConfig.substituted`

`EnvironmentConfig` records whether its values have been expanded, so
downstream layers can ask instead of guessing:

| How the config was built | `substituted` |
|---|---|
| `EnvironmentConfig.load(...)` / `.from_dict(...)` (default) | `True` |
| the same with `substitute_vars=False` | `False` |
| direct dataclass construction — `EnvironmentConfig(name=..., resources=...)` | `False` |

`EnvironmentAwareConfig` and `ConfigBindingResolver` read this and skip
their own pass when the environment has already been expanded. A
directly-constructed environment still gets substituted by them, so that
path is unchanged.

`substituted` is excluded from equality: two configs holding the same values
are the same environment regardless of which layer expanded them.

Call `substituted_view()` to get an expanded equivalent. It returns `self`
when the config is already substituted, and it never mutates the receiver —
so a caller holding raw refs on purpose keeps them even after a resolution
layer has read through the config.

```python
env = EnvironmentConfig.load("production")     # substituted=True
env.substituted_view() is env                  # True -- no second pass

raw = EnvironmentConfig(name="test", resources={...})   # substituted=False
view = raw.substituted_view()                  # a substituted copy
raw.substituted                                # still False
```

`substituted_view()` covers every field, `name` and `description` included,
because `load()` / `from_dict()` substitute the whole raw document before
constructing — a view covering less would set the same flag over a narrower
claim.

`merge()` keeps this uniform: merging a substituted config with an
unsubstituted one expands the unsubstituted side during the merge, rather
than producing a config whose single flag is wrong for half its values. That
pass reads the process environment, so a merge of two sides that *disagree*
can raise `RequiredEnvVarError` on an unset variable. Merging two sides that
agree touches no environment variables and cannot raise.

`substituted` records how a config was built, not what it currently holds.
Writing into `resources` or `settings` after construction does not update it,
and a downstream layer reading a stale `True` skips the pass those new values
needed. Build the config you want rather than amending one; if you must
amend, re-mark it with `dataclasses.replace(env, substituted=False)`.

### `InheritableConfigLoader`: provenance in the cache key

The loader resolves `extends:` by loading the parent with
`substitute_vars=False` and substituting the merged result once, at the end.
So one config can be produced in two forms, and the cache keys on
`(name, substitute_vars)` to keep them apart. A shared key would let either
form serve a request for the other, in both directions:

* load a child, then its parent — the parent comes back with raw `${VAR}`
  placeholders, expanded **zero** times;
* load a parent, then its child — the already-expanded parent is merged in
  and the result expanded again, **twice**: the `p${x}ss` → `pINJECTEDss`
  failure above, reached through a different door.

`clear_cache(name)` clears every variant stored under that name, so clearing
a config clears the config rather than one of its two forms.

The provenance record is the key here rather than a flag on the value
because the cached value is a bare `dict`, with nowhere to hang one. Same
rule, two mechanisms, chosen by what the cached thing is: an
`EnvironmentConfig` is an object and can carry its own answer; a `dict`
needs its container to remember.

### Serializing: `to_dict()` / `from_dict()`

`to_dict()` emits values as held — already expanded for a config built the
default way — and deliberately does **not** emit provenance, which would put
a hand-editable metadata key into what is otherwise a serialized environment
file. So the naive round-trip substitutes a second time:

```python
# WRONG for any value whose text contains ${...}
EnvironmentConfig.from_dict(env.to_dict())

# Correct
EnvironmentConfig.from_dict(env.to_dict(), substitute_vars=False)
```

Unlike the resolution layers, this composition is one the caller performs
and can spell correctly, which is why it is documented rather than
worked around.
<!-- --8<-- [end:substitute-once] -->

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

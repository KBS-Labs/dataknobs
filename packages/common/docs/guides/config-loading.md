# Shared YAML/JSON Config Loading

`dataknobs_common.config_loading` provides the helpers every dataknobs
config-file loader uses for the parse-and-validate chain. It
consolidates what was previously nine separate sites duplicating
``yaml.safe_load`` / ``json.load`` + extension probing + "must be a
dict" checks across `dataknobs_config`, `dataknobs_xization`,
`dataknobs_fsm`, `dataknobs_bots`, and `dataknobs_llm`.

## Helpers

| Helper | Role |
|---|---|
| `find_config_file(config_dir, name, *, extensions=..., allow_outside=False)` | Probe a directory for `name.{yaml,yml,json}` (or any custom extensions) and return the first existing path, or `None`. `name` must stay inside `config_dir` — see [Names Are Bounded by `config_dir`](#names-are-bounded-by-config_dir). |
| `load_yaml_or_json(path, *, require_dict=True, encoding="utf-8")` | Open a path and parse it according to its extension. Optionally requires a dict at the root. |
| `parse_yaml_or_json(data, *, format, source_name=None, require_dict=True)` | Parse already-loaded bytes or text content. Used by consumers that pull bytes from non-filesystem sources (e.g. `KnowledgeResourceBackend`). |

## Exception Hierarchy

All helper-raised errors subclass `ConfigLoadError`, so a consumer can
catch the base type and re-raise as its own error class:

```text
ConfigLoadError              (base)
├── ConfigParseError         (YAML/JSON parser failed)
├── ConfigPathEscapeError    (name addresses a file outside config_dir)
├── ConfigShapeError         (require_dict=True and root is not a dict)
├── ConfigUnsupportedFormatError  (extension or format hint not recognized)
└── ConfigYAMLNotInstalledError   (YAML payload requested, PyYAML missing)
```

`load_yaml_or_json` may also raise `FileNotFoundError` and `OSError`
for path-level I/O failures — those are stdlib types, not subclasses
of `ConfigLoadError`.

## The "Consumer Keeps Its Own Error Class" Pattern

Each migrated consumer wraps the helper in a small try/except so its
public error surface is unchanged:

```python
from dataknobs_common.config_loading import (
    ConfigLoadError, find_config_file, load_yaml_or_json,
)

class MyError(Exception):
    pass

class MyLoader:
    def load(self, name: str) -> dict[str, Any]:
        path = find_config_file(self.config_dir, name)
        if path is None:
            raise MyError(f"Config not found: {name} in {self.config_dir}")
        try:
            return load_yaml_or_json(path)
        except ConfigLoadError as e:
            raise MyError(str(e)) from e
        except OSError as e:
            raise MyError(f"Failed to read {path}: {e}") from e
```

This pattern is how all five consumer packages route through the
shared helper while keeping their existing public error types
(`InheritanceError`, `EnvironmentConfigError`,
`EnvironmentAwareConfigError`, `ValidationError` /
`ConfigFileNotFoundError`, `IngestionConfigError`, `ValueError` for
FSM and `FileSystemPromptLibrary`).

`RAGKnowledgeBase._load_kb_config_from_backend` (in
`dataknobs_bots.knowledge`) reuses `IngestionConfigError` from
`dataknobs_xization.ingestion` deliberately, so the local-directory
and backend-driven ingest paths surface the same error class.

## PyYAML is Optional

`dataknobs-common` does **not** declare PyYAML as a hard dependency.
The YAML parser is imported lazily inside `parse_yaml_or_json` (and
indirectly from `load_yaml_or_json`). Consumers that need YAML support
must install PyYAML themselves; if they pass a `.yaml` / `.yml` file
without PyYAML installed, the helper raises
`ConfigYAMLNotInstalledError` with a clear remediation message.

JSON support is built-in (stdlib `json`).

## Environment Variable Substitution Is the Caller's Concern

The helpers deliberately do NOT apply `${VAR}` substitution. Different
consumers have different substitution flag profiles
(`expand_user_paths`, `type_coerce`, `substitute_keys`), and not every
consumer wants substitution at load time at all (`EnvironmentAwareConfig`
deliberately defers substitution to its `resolve_for_build` step for
late binding).

If you want substitution, wire `dataknobs_config.substitute_env_vars`
in at the consumer level after `load_yaml_or_json` returns. See
`docs/packages/config/environment-variables.md` for the substitution
reference.

## Names Are Bounded by `config_dir`

`find_config_file` composes `config_dir / name`, and the result has to
stay inside `config_dir`. A name may address a **subdirectory** — that
is how a layout convention is expressed — but a name that *lands*
outside raises `ConfigPathEscapeError`. Two spellings get there: a `..`
segment walks out of the directory, and an absolute name discards
`config_dir` entirely, because that is what `Path("/base") / "/etc/x"`
evaluates to. Neither is rejected on sight — containment is judged on
where the composed path lands, so `sub/../x` and an absolute name
pointing back inside `config_dir` are both allowed:

```python
find_config_file(config_dir, "domains/child")   # fine: inside config_dir
find_config_file(config_dir, "../../etc/x")     # ConfigPathEscapeError
find_config_file(config_dir, "/etc/x")          # ConfigPathEscapeError
```

The check runs **before** any file is probed, so an escaping name fails
identically whether or not the file it names exists — the guard is not
also an existence oracle.

This matters because the `name` reaching this helper is often not a
literal the calling code wrote. It can be an `extends:` value read out
of a config file, an environment name read from the process
environment, or the output of a consumer-supplied name resolver. The
loader recursing into those names never returns to the caller in
between, so bounding them at the join is the only place that covers all
of them.

A caller that deliberately addresses a tree outside its search
directory passes `allow_outside=True`; the escaping name is then logged
at WARNING and probed anyway, so the bypass is auditable rather than
silent, and a contained name logs nothing. The three `dataknobs-config`
loaders forward the flag rather than hardcoding the default, so a
deployment reaches it without dropping to this helper:

```python
InheritableConfigLoader(config_dir, allow_outside=True)
EnvironmentConfig.load(env, config_dir, allow_outside=True)
EnvironmentAwareConfig.load_app(app, app_dir, env_dir, allow_outside=True)
```

Every one of them defaults to `False`. Weigh the two environment-driven
loaders separately before opting out: their name can come from
`DATAKNOBS_ENVIRONMENT` or `ENVIRONMENT`, so the flag widens what a
process environment variable can address.

<!-- --8<-- [start:safe-join-primitive] -->
### `safe_join`, the primitive underneath

The containment test is `dataknobs_common.paths.safe_join`, exported
from the package root and usable directly by any code that composes a
path out of a name it did not write:

```python
from dataknobs_common import safe_join

target = safe_join(knowledge_dir, domain_id, resource_path)
if target is None:
    raise ValueError("resource path resolves outside the knowledge directory")
```

It returns the joined path with `.` and `..` segments collapsed, or
`None` when the result lands outside the base. Containment is judged on
where the path **lands**, not on how it was spelled: an interior
`a/../b` is contained, and so is an absolute part that points back
inside the base.

The check is lexical — no filesystem access — so it is safe to run on
an event loop and works on a path that does not exist yet, which is the
case whenever the composed path is about to be written. The trade-off
is that symlinks are not followed: a symlinked file inside the base is
treated as contained, deliberately — a Kubernetes ConfigMap mount is
built out of symlinks. Two consequences that belong together:

- **Use the path it returns**, not the name you passed in. The returned
  path is the collapsed one. Re-composing the name as written reopens
  the gap the guard just closed, because a symlinked subdirectory plus
  a `..` resolves through the link's target and lands outside the base.
- A write that must not be redirected through such a symlink wants a
  `Path.resolve()` check of its own as well, off the loop.
<!-- --8<-- [end:safe-join-primitive] -->


## Custom Extension Sets

`find_config_file` accepts a custom `extensions` tuple:

```python
from dataknobs_common.config_loading import find_config_file

# Only consider .json (skip the YAML fallbacks):
match = find_config_file(config_dir, "production", extensions=(".json",))

# Add .toml as a future-proofing escape hatch (parser not built in):
match = find_config_file(
    config_dir, "production",
    extensions=(".yaml", ".yml", ".json", ".toml"),
)
```

The extensions are tried in the order given. A leading dot is added
automatically if missing (`"yaml"` → `".yaml"`).

## Empty Files

Empty YAML and empty JSON behave differently:

- Empty YAML files parse to `None` (PyYAML's `safe_load` of an empty
  string returns `None`). With `require_dict=False` they pass through
  as `None`; with the default `require_dict=True` they raise
  `ConfigShapeError`.
- Empty JSON files raise `ConfigParseError` regardless of
  `require_dict` (stdlib `json.loads("")` raises `JSONDecodeError`).

Consumers that want to treat empty files as `{}` can apply that
coercion at the call site (e.g., `data if data else {}`).

## When to Use `parse_yaml_or_json` vs `load_yaml_or_json`

Use `load_yaml_or_json` when you have a filesystem path. It handles
the open/read for you, picks YAML vs JSON from the extension, and
forwards to `parse_yaml_or_json` internally.

Use `parse_yaml_or_json` directly when the bytes already live in
memory — for example, when reading from a
`KnowledgeResourceBackend.get_file()` which returns raw bytes from
S3, an in-memory store, or a filesystem-backed implementation. The
consumer chooses the format explicitly via the keyword-only `format`
argument since there's no path extension to infer from.

## Migration Notes for Out-of-Tree Consumers

These helpers are net-additive. If you already had your own
yaml-or-json loader, switching to `dataknobs_common.config_loading`
involves:

1. Replace your inline parse-and-validate with `load_yaml_or_json`.
2. Wrap the call in a `try/except ConfigLoadError` block to preserve
   your error class.
3. If you also need extension probing, replace any
   `for ext in (".yaml", ".yml", ".json")` loop with
   `find_config_file(config_dir, name)`.

Each migration is behavior-preserving by construction so long as the
wrap re-raises with the same error class.

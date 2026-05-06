# Config Loading

`dataknobs_common.config_loading` consolidates the YAML/JSON
file→dict and bytes→dict parse-and-validate chain that every
dataknobs config-file loader uses.

## Why This Exists

Before this module landed, nine sites across five packages each
duplicated the same chain — `yaml.safe_load` / `json.load`, extension
probing, "must be a dict" validation — wrapping the failures in a
different error class:

| Consumer | Error class |
|---|---|
| `InheritableConfigLoader._load_file` | `InheritanceError` |
| `EnvironmentConfig._load_file` | `EnvironmentConfigError` |
| `EnvironmentAwareConfig._load_file` | `EnvironmentAwareConfigError` |
| `Config._load_file` / `Config._load_referenced_file` | `ValidationError` / `ConfigFileNotFoundError` |
| `KnowledgeBaseConfig._load_file` | `IngestionConfigError` |
| `FSMConfigLoader._load_file` | `ValueError` (FSM-internal surface) |
| `RAGKnowledgeBase._load_kb_config_from_backend` | `IngestionConfigError`* |
| `FileSystemPromptLibrary._load_file` | `ValueError` |

\* `RAGKnowledgeBase` lives in `dataknobs_bots.knowledge` but
deliberately reuses `IngestionConfigError` from
`dataknobs_xization.ingestion` so the local-directory and
backend-driven ingest paths surface the same error class.

The duplication had already produced one cross-cutting consistency
bug (env-config silently skipping env-var substitution that every
sibling loader applied). This module is the preventative refactor:
the chain now lives in one place; consumers wrap it to preserve their
public error class.

## Helpers

```python
from dataknobs_common.config_loading import (
    find_config_file,
    load_yaml_or_json,
    parse_yaml_or_json,
)
```

### `find_config_file(config_dir, name, *, extensions=...)`

Probes a directory for `name.{yaml,yml,json}` (or any custom set of
extensions). Returns the first existing path or `None`.

```python
from pathlib import Path
from dataknobs_common.config_loading import find_config_file

path = find_config_file(Path("config/environments"), "production")
# -> Path("config/environments/production.yaml") if it exists, else None
```

### `load_yaml_or_json(path, *, require_dict=True, encoding="utf-8")`

Opens a path, picks the parser from its extension, and validates the
shape. Returns the parsed payload.

```python
from dataknobs_common.config_loading import load_yaml_or_json

data = load_yaml_or_json("config/environments/production.yaml")
# -> dict[str, Any]
```

Pass `require_dict=False` to tolerate list / scalar / `None` roots —
this is what `Config._load_file` uses, since referenced files may
contain non-dict payloads.

Empty files behave differently between the two formats:

- Empty YAML parses to `None` (and passes through with
  `require_dict=False`).
- Empty JSON raises `ConfigParseError` regardless of `require_dict`
  (stdlib `json.loads("")` is a parse error).

Consumers that want both empty-file forms to collapse to `{}` apply
`data if data else {}` at the call site.

### `parse_yaml_or_json(data, *, format, source_name=None, require_dict=True)`

Parses already-in-memory bytes or text. Used by consumers that pull
content from non-filesystem sources (e.g. an S3-backed
`KnowledgeResourceBackend`):

```python
from dataknobs_common.config_loading import parse_yaml_or_json

raw_bytes = await backend.get_file(domain_id, "knowledge_base.yaml")
data = parse_yaml_or_json(
    raw_bytes,
    format="yaml",
    source_name=f"{domain_id}/knowledge_base.yaml",
)
```

## Exception Hierarchy

```text
ConfigLoadError                  (base)
├── ConfigParseError             (YAML/JSON parser failed)
├── ConfigShapeError             (require_dict=True and root is not a dict)
├── ConfigUnsupportedFormatError (extension or format hint not recognized)
└── ConfigYAMLNotInstalledError  (YAML payload requested, PyYAML not installed)
```

`load_yaml_or_json` can also raise `FileNotFoundError` and `OSError`
for path-level I/O failures — those are stdlib types, not subclasses
of `ConfigLoadError`.

## The "Consumer Wraps to Keep Its Own Error Class" Pattern

Every migrated consumer follows the same shape:

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
            raise MyError(f"Not found: {name} in {self.config_dir}")
        try:
            return load_yaml_or_json(path)
        except ConfigLoadError as e:
            raise MyError(str(e)) from e
        except OSError as e:
            raise MyError(f"Read failed: {path}: {e}") from e
```

This is how `dataknobs-config`, `dataknobs-xization`, `dataknobs-fsm`,
`dataknobs-bots`, and `dataknobs-llm` all route through the shared
helper while keeping their existing public error types unchanged.

## PyYAML Is Optional

`dataknobs-common` does **not** declare PyYAML as a hard dependency.
`yaml` is imported lazily inside `parse_yaml_or_json`. If a consumer
calls `load_yaml_or_json` on a `.yaml` / `.yml` file without PyYAML
installed, the helper raises `ConfigYAMLNotInstalledError` with an
install hint. JSON support is stdlib-built-in.

## Substitution Is the Caller's Concern

The helpers deliberately do NOT apply `${VAR}` substitution. Different
consumers want different substitution flag profiles, and some
(`EnvironmentAwareConfig`) defer substitution entirely to a
late-binding step. If you want substitution, call
`dataknobs_config.substitute_env_vars` on the result.

See [Environment Variables](../config/environment-variables.md) for
the substitution helper reference.

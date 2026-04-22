# Postgres Connection Configuration

`dataknobs_common.normalize_postgres_connection_config` is the single
source of truth for how every dataknobs postgres-using construct parses
a connection configuration. It accepts every input shape the framework
has historically supported and emits a canonical dict that downstream
consumers read uniformly.

## Consumers

A single config dict feeds all of these:

- `PgVectorStore` (`dataknobs_data.vector.stores.pgvector`)
- `SyncPostgresDatabase` / `AsyncPostgresDatabase`
  (`dataknobs_data.backends.postgres`)
- `PostgresPoolConfig` (`dataknobs_data.pooling.postgres`)
- `PostgresEventBus` (`dataknobs_common.events.postgres`)

## Accepted Input Shapes

### 1. `connection_string`

```python
config = {
    "connection_string": "postgresql://user:pass@host:5432/mydb",
}
```

The dialect prefix `postgresql+asyncpg://` is stripped to
`postgresql://` so asyncpg, psycopg2, and sqlalchemy-style URIs are all
accepted. The URL supplies the base values for every canonical key;
individual keys in the same config (see #2) override the URL on a
per-field basis.

### 2. Individual keys

```python
config = {
    "host": "db.example.com",
    "port": 5433,
    "database": "mydb",
    "user": "admin",
    "password": "secret",
}
```

`user` and `password` are URL-encoded when a canonical
`connection_string` is synthesized, so values containing `@`, `/`, or
`:` (common in secrets-manager output) produce a valid URI. `host` and
`database` are rejected if they contain `@`, `/`, or whitespace —
URL-encoding them would distort legitimate values, and the intent is
to surface misconfiguration loudly rather than produce a malformed
URI.

IPv6 literal host addresses (e.g. `::1`, `2001:db8::1`) are not
supported via individual keys because the synthesizer does not add
the `[...]` brackets that a bare-IPv6 authority requires. Pass them
through `connection_string` instead, pre-bracketed:
`postgresql://u:p@[::1]:5432/db`.

Individual keys can also be mixed with `connection_string` to
override specific fields — see Precedence below.

### 3. `DATABASE_URL` env var

Used only when no `connection_string` **and** no individual keys are
present in config:

```bash
export DATABASE_URL="postgresql://user:pass@host:5432/mydb"
```

```python
# The normalizer reads DATABASE_URL and returns canonical keys.
config = {}
```

### 4. `POSTGRES_*` env vars

Fill gaps left by config and by `DATABASE_URL`. Individual config keys
always win over these env fallbacks:

```bash
export POSTGRES_HOST=db.example.com
export POSTGRES_PORT=5433
export POSTGRES_DB=mydb
export POSTGRES_USER=admin
export POSTGRES_PASSWORD=secret
```

```python
config = {}  # env supplies the connection
```

### 5. `.env` / `.project_vars` files

Values from a `.env` or `.project_vars` file in the current working
directory (or a parent directory) are used as an additional fallback
layer — they do **not** override `os.environ`, they fill remaining
gaps. This restores the behavior of the retired
`DotenvPostgresConnector` for developers who keep local credentials
in those files.

Requires `python-dotenv` to be installed. If absent, this layer is
silently skipped. Tests that need strict env isolation can disable
the layer with `load_dotenv=False`.

## Precedence

In order of precedence, per canonical key (`host`, `port`, `database`,
`user`, `password`):

1. **Individual key in `config`** — always wins. A caller passing
   `{"host": "myhost"}` expects `"myhost"`, and `DATABASE_URL` in the
   shell must not silently override it.
2. **Value parsed from `config["connection_string"]`** — supplies
   the base for keys not set individually. A caller passing
   `{"connection_string": "postgresql://u:p@h/db", "database":
   "override"}` connects to `override`, not `db`.
3. **`DATABASE_URL` env var** — used only when no
   `connection_string` is in config and no individual keys in config
   provide any value. Individual keys in config suppress it so the
   shell env cannot silently override caller intent.
4. **`POSTGRES_HOST` / `POSTGRES_PORT` / `POSTGRES_DB` /
   `POSTGRES_USER` / `POSTGRES_PASSWORD`** env vars — fill remaining
   gaps.
5. **`.env` / `.project_vars` file values** — fill gaps that
   `os.environ` did not.
6. **Defaults**: `localhost:5432/postgres` with user `postgres` and
   empty password. When defaults are used alongside explicit config
   or a connection_string, the normalizer logs a WARNING naming the
   defaulted fields so partial misconfiguration surfaces loudly.

## API Reference

```python
from dataknobs_common import normalize_postgres_connection_config

normalized = normalize_postgres_connection_config(
    config, require=True,
)
```

**Parameters:**

- `config` (`dict | None`): User config dict. Extra backend-specific
  keys are preserved verbatim. `None` is treated as an empty dict.
- `require` (`bool`, keyword-only, default `True`): When `True`, raise
  `ConfigurationError` if no postgres connection can be resolved.
  When `False`, return `None` instead — lets optional-backend
  consumers treat "no postgres configured" as a soft signal.
- `load_dotenv` (`bool`, keyword-only, default `True`): When `True`,
  read `.env` / `.project_vars` files as an additional env fallback
  layer (see Accepted Input Shapes #5). Set `False` in tests that
  require strict env isolation.

**Returns:**

A new dict containing every key from the input plus canonical
`connection_string`, `host`, `port` (coerced to `int`), `database`,
`user`, and `password`. Returns `None` only when `require=False` and
no connection is resolvable.

**Raises:**

- `ConfigurationError`: When `require=True` and no connection can be
  resolved from any of the accepted input shapes.
- `ValueError`: When `host` or `database` contain shell-unsafe
  characters (`@`, `/`, whitespace) that would produce a malformed or
  misrouted URI.

## Examples

### Migrating from `connection_string`-only callers

Before (only a DSN string worked):

```python
from dataknobs_common.events.postgres import PostgresEventBus

bus = PostgresEventBus(
    connection_string="postgresql://user:pass@host:5432/db",
)
```

After (same call still works, plus the unified config dict):

```python
bus = PostgresEventBus(
    config={
        "host": "host",
        "port": 5432,
        "database": "db",
        "user": "user",
        "password": "pass",
    },
)
```

Or via env vars only:

```bash
export POSTGRES_HOST=host
export POSTGRES_USER=user
export POSTGRES_PASSWORD=pass
export POSTGRES_DB=db
```

```python
bus = PostgresEventBus(config={})
```

### Partial config + env fallback

```bash
export POSTGRES_PASSWORD=secret
```

```python
# host/database from config; password from env
config = {"host": "db.example.com", "database": "mydb"}
normalized = normalize_postgres_connection_config(config)
# normalized["password"] == "secret"
```

### Overriding one field of a shared connection string

```python
# Shared URL for the team, but this test run targets a different db.
config = {
    "connection_string": "postgresql://u:p@host:5432/primary_db",
    "database": "test_db",
}
normalized = normalize_postgres_connection_config(config)
assert normalized["host"] == "host"          # from the URL
assert normalized["database"] == "test_db"   # individual key wins
```

### Optional backend (no postgres configured)

```python
normalized = normalize_postgres_connection_config({}, require=False)
assert normalized is None
# caller can fall back to a non-postgres backend
```

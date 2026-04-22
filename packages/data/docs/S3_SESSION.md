# S3 Session Configuration

`S3SessionConfig` is the canonical helper for constructing
`boto3` / `aioboto3` S3 clients across all dataknobs S3-using
constructs. It owns the rules for region resolution, `endpoint_url`
handling, credential passthrough, and retry / pool defaults so every
S3 consumer sees identical defaults.

**Routes through this layer:**

- `dataknobs_data.backends.s3.SyncS3Database` (sync `boto3`) — calls
  `create_boto3_s3_client` directly with its `S3SessionConfig`.
- `dataknobs_data.backends.s3_async.AsyncS3Database` (async
  `aioboto3`) — passes its `S3PoolConfig` to `create_aioboto3_session`,
  which projects onto `S3SessionConfig` internally via
  `S3PoolConfig.to_session_config()`.
- `dataknobs_bots.knowledge.storage.s3.S3KnowledgeBackend` (sync
  `boto3`) — calls `create_boto3_s3_client` directly.
- `dataknobs_data.pooling.s3.validate_s3_session` (async client used
  for pool validation) — reuses the same kwarg-shaping helper as
  `S3SessionConfig.to_client_kwargs()`.

## Accepted Input Shapes

`S3SessionConfig.from_dict(config)` accepts any of these keys
(unrecognized keys are ignored):

| Concern | Canonical key | Aliases | Notes |
|---|---|---|---|
| Region | `region_name` | `region` | `region_name` wins when both present. |
| Endpoint | `endpoint_url` | — | Only included in client kwargs when truthy. |
| Access key | `aws_access_key_id` | `access_key_id` | |
| Secret key | `aws_secret_access_key` | `secret_access_key` | |
| Session token | `aws_session_token` | `session_token` | |
| Connection pool | `max_pool_connections` | `max_workers` | Default `10`. |
| Retry attempts | `max_attempts` | `max_retries` | Default `3`. |
| Retry mode | `retry_mode` | — | Default `"standard"`. |
| Extra client kwargs | `extra_client_kwargs` | — | Passthrough dict. Applied last. |

Missing keys default to `None` so boto's resolution chain decides
the value at client-construction time.

## Default Region Resolution

When `region_name` is unset (which is now the default for all
dataknobs S3 constructs), `botocore` resolves the region in this
order:

1. `AWS_DEFAULT_REGION` environment variable
2. `~/.aws/config` `[default]` (or `[profile <name>]`) `region` entry
3. EC2 / ECS instance metadata (IMDS), when running in AWS
4. `us-east-1` as boto's terminal fallback

> **Note:** `botocore` reads `AWS_DEFAULT_REGION` for region
> resolution, not `AWS_REGION`. Consumers that previously relied on
> `AWS_REGION` should set `AWS_DEFAULT_REGION` instead.

## SSL for `http://` Endpoints

When `endpoint_url` starts with `http://` (LocalStack, MinIO, dev
S3-compatible servers), `use_ssl=False` is added to the client
kwargs automatically so `botocore` doesn't attempt TLS on a
plain-HTTP port. `https://` endpoints leave `use_ssl` unset so
boto's default (`True`) applies. Callers can override either case
via `extra_client_kwargs={"use_ssl": ...}`.

## Examples

### Sync — `boto3` client straight from a dict

```python
from dataknobs_data.pooling.s3 import create_boto3_s3_client

# Defers entirely to boto's default chain.
s3 = create_boto3_s3_client()

# Explicit region; everything else still defers to the chain.
s3 = create_boto3_s3_client({"region_name": "eu-west-1"})

# Full configuration.
s3 = create_boto3_s3_client(
    {
        "region": "us-west-2",
        "endpoint_url": "http://localhost:4566",  # LocalStack
        "aws_access_key_id": "test",
        "aws_secret_access_key": "test",
        "max_pool_connections": 25,
        "max_attempts": 5,
    }
)
```

### Async — `aioboto3` session via `S3PoolConfig`

```python
from dataknobs_data.pooling.s3 import (
    S3PoolConfig,
    create_aioboto3_session,
)

pool_cfg = S3PoolConfig.from_dict(
    {"bucket": "my-bucket", "region": "eu-west-1"}
)
session = await create_aioboto3_session(pool_cfg)
async with session.client("s3") as s3:
    await s3.head_bucket(Bucket=pool_cfg.bucket)
```

### Sharing one config across multiple consumers

```python
from dataknobs_data.pooling.s3 import S3SessionConfig
from dataknobs_bots.knowledge.storage.s3 import S3KnowledgeBackend

shared = S3SessionConfig.from_dict(
    {"region": "eu-west-1", "endpoint_url": "http://localhost:4566"}
)
backend_a = S3KnowledgeBackend(
    bucket="kb-a", session_config=shared
)
backend_b = S3KnowledgeBackend(
    bucket="kb-b", session_config=shared
)
```

### Same config for sync + async

Because `S3SessionConfig.from_dict` and `S3PoolConfig.from_dict`
both accept `region` and `region_name`, one config dict feeds both
constructs without rename.

```python
cfg = {"bucket": "my-bucket", "region": "eu-west-1"}

from dataknobs_data.backends.s3 import SyncS3Database
from dataknobs_data.backends.s3_async import AsyncS3Database

sync_db = SyncS3Database(cfg)
async_db = AsyncS3Database(cfg)
```

## API Reference

### `S3SessionConfig`

Frozen dataclass holding the normalized configuration for an S3
session.

| Field | Type | Default | Notes |
|---|---|---|---|
| `region_name` | `str \| None` | `None` | Boto's chain resolves when unset. |
| `endpoint_url` | `str \| None` | `None` | Omitted from kwargs when unset. |
| `aws_access_key_id` | `str \| None` | `None` | |
| `aws_secret_access_key` | `str \| None` | `None` | |
| `aws_session_token` | `str \| None` | `None` | |
| `max_pool_connections` | `int` | `10` | |
| `max_attempts` | `int` | `3` | |
| `retry_mode` | `str` | `"standard"` | |
| `extra_client_kwargs` | `dict[str, Any]` | `{}` | Passthrough. |

Methods:

- `from_dict(config)` — normalize a config dict; accepts the alias
  keys listed above.
- `to_boto_config_kwargs()` — kwargs for
  `botocore.config.Config(...)`.
- `to_client_kwargs()` — kwargs for `boto3.client("s3", ...)` /
  `session.client("s3", ...)`. Omits unset optional fields.

### `create_boto3_s3_client(config=None)`

Sync factory. Accepts an `S3SessionConfig`, a raw dict (normalized
internally), or `None` (full default chain). Returns a configured
`boto3` S3 client.

### `create_aioboto3_session(config)`

Async factory. Accepts an `S3PoolConfig` (existing async-pool
callers) or `S3SessionConfig` (new shared shape). Returns an
`aioboto3.Session`. Note: `endpoint_url` is per-client in aioboto3,
so it is applied at `session.client("s3", ...)` time, not at
session construction.

### `S3PoolConfig`

Existing pool-manager configuration. Adds:

- `region` accepted as alias for `region_name` in `from_dict`.
- `to_session_config()` — projects to the shared `S3SessionConfig`
  shape (drops `bucket`/`prefix`).

### `validate_s3_session(session, config)`

Async pool validation. Omits `endpoint_url` from client kwargs when
none is configured (no longer passes empty-string overrides).

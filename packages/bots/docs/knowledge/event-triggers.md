# Event triggers for knowledge backends

A `KnowledgeResourceBackend` writes three classes of keys: consumer
content uploaded via `put_file`, the DK-managed `_metadata.json` per
knowledge base, and (in snapshot mode) the per-version
`_snapshots/<version>.json` records under each KB. External event
sources (S3 → EventBridge / SQS / SNS / Lambda; filesystem inotify;
GCS Pub/Sub; etc.) MUST filter to the content subtree so they don't
retrigger on the DK-managed state writes the ingestion manager emits
during ingest.

This page describes the filter contract and shows how to wire it at
each common event source.

## Layout

For every in-tree backend, each knowledge base lives under a
backend-specific prefix:

```
{prefix}/
    {domain_id}/
        content/              # consumer-controlled (put_file)
            file1.md
            subdir/file2.json
        _metadata.json        # DK-managed state
        _snapshots/           # DK-managed state (snapshot mode)
            <version>.json
```

Where `{prefix}` is:

| Backend | Prefix |
|---|---|
| `S3KnowledgeBackend` | the constructor's `prefix` arg (default `"knowledge/"`), under `s3://{bucket}/`. |
| `FileKnowledgeBackend` | the constructor's `base_path`. |
| `InMemoryKnowledgeBackend` | no on-backend prefix; the layout is conceptual only. |

## Failure mode: the positive feedback loop

If an external event source fires on *every* write to the backend —
not just consumer content uploads — it will fire on the
`_metadata.json` and `_snapshots/<version>.json` writes that the
ingestion manager performs as part of finishing an ingest. Each of
those state writes appears, to the event source, indistinguishable
from a fresh consumer upload, so the trigger fires again. The next
fire drives another ingest, which writes state again, and so on. This
positive feedback loop saturates the trigger's downstream queue (SQS,
SNS subscribers, Lambda invocations) until the source is reconfigured
or rate-limited externally.

`classify_key` and `key_pattern` are the affordances that let
consumers wire the right filter without re-reading the layout
contract or hard-coding constants the backend declares privately.

## The helper API

Every in-tree backend (and any out-of-tree backend mixing in
`KnowledgeResourceBackendMixin`) exposes two methods:

```python
from dataknobs_bots.knowledge.storage import KnowledgeKeyKind

# Source-level filter — forward verbatim to the event source.
pattern = backend.key_pattern(KnowledgeKeyKind.CONTENT)         # all KBs
pattern = backend.key_pattern(
    KnowledgeKeyKind.CONTENT, domain_id="acme",
)                                                                # one KB

# Per-event classification — for sources that can't pattern-match.
kind = backend.classify_key(received_key)
if kind is not KnowledgeKeyKind.CONTENT:
    return  # skip state writes
```

`KnowledgeKeyKind` has four members:

| Member | Meaning |
|---|---|
| `CONTENT` | Consumer-controlled, written by `put_file()`. External event triggers SHOULD subscribe to this kind. |
| `METADATA` | DK-managed `_metadata.json`. External event triggers MUST NOT subscribe unless explicitly auditing state changes. |
| `SNAPSHOT` | DK-managed `_snapshots/<version>.json` (snapshot mode). External event triggers MUST NOT subscribe unless explicitly archiving snapshot history. |
| `UNKNOWN` | Key did not match any declared layout segment. Surfaces as a defensive default for future protocol extensions. |

Prefer `key_pattern` for source-level filtering when the event source
supports patterns — filtering upstream avoids paying the
message-receive cost for state writes. Fall back to `classify_key`
when patterns aren't available.

## Per-source wiring

### S3 → EventBridge (`s3.event.detail.object.key` wildcard)

```python
from dataknobs_bots.knowledge.storage import KnowledgeKeyKind

pattern = s3_backend.key_pattern(KnowledgeKeyKind.CONTENT)
# e.g. "knowledge/*/content/*"
```

In CDK / CloudFormation:

```python
events.EventPattern(
    source=["aws.s3"],
    detail_type=["Object Created"],
    detail={
        "bucket": {"name": [s3_backend.bucket]},
        "object": {"key": [{"wildcard": pattern}]},
    },
)
```

Single-domain scope (e.g. routing per-tenant):

```python
pattern = s3_backend.key_pattern(
    KnowledgeKeyKind.CONTENT, domain_id="acme",
)
# e.g. "knowledge/acme/content/*"
```

### S3 → bucket notification (SQS / SNS / Lambda direct)

The bucket-notification filter syntax doesn't support full wildcards —
only `prefix` + `suffix`. Derive both from the helper output:

```python
pattern = s3_backend.key_pattern(KnowledgeKeyKind.CONTENT)
# Strip the trailing "*" and use the literal segment as the prefix.
prefix = pattern.rstrip("*")
# e.g. "knowledge/*/content/" — note bucket-notification does NOT
# expand "*", so this exact form will only catch top-level files
# inside content/. Use EventBridge above for true wildcard matching.
```

For bucket-notification consumers, the practical pattern is to
hardcode the per-KB content prefix once at deploy time:

```python
prefix = s3_backend.key_pattern(
    KnowledgeKeyKind.CONTENT, domain_id="acme",
).rstrip("*")
# e.g. "knowledge/acme/content/"
```

then configure a bucket-notification rule with that prefix per
knowledge base.

### Filesystem inotify / polling watcher

```python
from pathlib import Path, PurePath

pattern = file_backend.key_pattern(KnowledgeKeyKind.CONTENT)
# e.g. "/srv/kb/*/content/**"

# pathlib.Path.glob honors the pattern directly.
content_files = Path("/").glob(pattern.lstrip("/"))

# For per-event filtering with an inotify-style wrapper:
for event in watcher.events():
    if file_backend.classify_key(event.path) is KnowledgeKeyKind.CONTENT:
        handle(event)
```

`PurePath.match` honors the same glob shape if a per-event match
helper is preferred.

### GCS Pub/Sub object-name filter

GCS object-name filters use a comparable wildcard syntax. Forward the
helper output as the filter expression; the per-source recipe
otherwise mirrors S3 → EventBridge.

### Generic fallback (no pattern support)

When the event source can't pattern-match (or when the consumer
prefers post-receive classification), accept every event and branch
on `classify_key`:

```python
from dataknobs_bots.knowledge.storage import KnowledgeKeyKind

async def handle_event(received_key: str) -> None:
    kind = backend.classify_key(received_key)
    if kind is KnowledgeKeyKind.CONTENT:
        await dispatch_to_ingest(received_key)
    elif kind in {KnowledgeKeyKind.METADATA, KnowledgeKeyKind.SNAPSHOT}:
        # Silently drop — DK-managed state write.
        return
    else:
        # KnowledgeKeyKind.UNKNOWN — defensive log; the layout may
        # have been extended in a newer backend version.
        logger.warning("Unrecognized knowledge-backend key: %s", received_key)
```

This pattern is the right fallback regardless of source — even for
sources that DO support patterns, a `classify_key` check at the
handler is a useful defense-in-depth assertion that the configured
filter is correctly excluding state writes.

## Out-of-tree backends

Any backend mixing in `KnowledgeResourceBackendMixin` inherits the
default `classify_key` implementation, which works against the
canonical `METADATA_FILE` / `CONTENT_DIR` / `SNAPSHOTS_DIR` constants
declared on the mixin. A third-party backend honoring the documented
layout gets correct classification for free.

`key_pattern` is per-backend because the pattern dialect depends on
the transport. Backends with no pattern-matching event source (the
in-memory backend, for example) return `""` — the method exists for
protocol symmetry so consumer code can call it uniformly across every
backend.

## See also

- [Knowledge Base Ingestion Guide](ingestion-guide.md) — the upstream
  ingest paths that drive the writes filtered on this page.
- [`IngestOrchestrator`](orchestrator.md) — the subscriber-side
  primitive most external triggers ultimately route to.

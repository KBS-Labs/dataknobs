"""Knowledge-layer event topic constants and consumer-extensibility helpers.

The four topic constants below are the canonical names knowledge-layer
adopters publish to via the in-process callback-registry substrate and
(via ``also_publish_to``) any composed event bus. The names follow the
``<subsystem>:<operation>:<phase>`` convention.

The :class:`KnowledgeTriggerPayload` ``TypedDict`` documents the shape of
the payloads the ingest orchestrator consumes. The fields are
intentionally not enforced at the publication boundary — the payload type
is ``dict[str, Any]`` end-to-end. The ``TypedDict`` is documentation:
consumer code *building* a payload imports it for editor completion and
type narrowing; consumer code *reading* a payload still works against
``dict[str, Any]`` and reads fields via ``payload.get(...)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Final, NotRequired, TypedDict

__all__ = [
    "INGEST_DOMAIN_END",
    "INGEST_DOMAIN_START",
    "INGEST_METADATA_WRITE",
    "INGEST_SNAPSHOT_WRITE",
    "KnowledgeTriggerPayload",
    "TenantFilteredCallback",
]


# --- Topic constants -------------------------------------------------- #

INGEST_DOMAIN_START: Final[str] = "ingest:domain:start"
"""Published by the ingestion manager when an ingest run begins.

Payload keys: ``domain_id`` (required), ``tenant_id`` (present iff the
manager is tenant-bound), ``started_at`` (ISO-8601 timestamp).
"""

INGEST_DOMAIN_END: Final[str] = "ingest:domain:end"
"""Published by the ingestion manager when an ingest run completes.

Payload keys: ``domain_id`` (required), ``tenant_id`` (present iff the
manager is tenant-bound), ``files_processed``, ``chunks_created``,
``files_deleted``, ``status`` (``"completed"`` / ``"failed"``),
``completed_at`` (ISO-8601 timestamp). Fires on every run exit
(success or failure).
"""

INGEST_METADATA_WRITE: Final[str] = "ingest:metadata:write"
"""Fired by a knowledge backend after every metadata state write.

Payload keys: ``domain_id`` (required), ``key`` (the metadata key
written), ``kind`` (always :attr:`KnowledgeKeyKind.METADATA`),
``byte_size``.
"""

INGEST_SNAPSHOT_WRITE: Final[str] = "ingest:snapshot:write"
"""Fired by a knowledge backend after every snapshot state write.

Payload keys: ``domain_id`` (required), ``key`` (the snapshot key
written), ``kind`` (always :attr:`KnowledgeKeyKind.SNAPSHOT`),
``byte_size``.
"""


# --- Trigger payload shape (documentation TypedDict) ----------------- #


class KnowledgeTriggerPayload(TypedDict):
    """Canonical shape of the payload the ingest orchestrator consumes.

    Field semantics:
        domain_id: knowledge-base domain to ingest (required).
        tenant_id: tenant binding for multi-tenant deployments
            (optional; absent = single-tenant).
        key: originating backend key for the trigger event (optional;
            S3-sourced triggers populate it from the object key,
            filesystem-watcher triggers populate it from the inotify
            path, pure cron / manual triggers omit). Consumers
            receiving a payload SHOULD call
            ``backend.classify_key(payload["key"])`` and skip
            non-``CONTENT`` keys.
        since_version: incremental-ingest cursor (optional).
        force_full: bypass change detection (optional; default False).
        last_version: consumer-provided snapshot version override
            (optional).

    The ``TypedDict`` is documentation. The wire format is
    ``dict[str, Any]``; the orchestrator reads fields via
    ``payload.get("key")``-style access, not ``payload["key"]``.
    """

    domain_id: str
    tenant_id: NotRequired[str]
    key: NotRequired[str]
    since_version: NotRequired[str]
    force_full: NotRequired[bool]
    last_version: NotRequired[str]


# --- Consumer-extensibility — tenant filter adapter ------------------ #


class TenantFilteredCallback:
    """Wrap an event callback and short-circuit on tenant mismatch.

    Construct with the inner callback plus the ``tenant_id`` to filter
    on. On every fire the adapter inspects ``event.get("tenant_id")``
    and invokes ``inner(event)`` only when the value equals the
    constructed ``tenant_id``. Events without a ``tenant_id`` key
    (single-tenant payloads) are dropped.

    Use case: a per-tenant consumer registers one callback on the
    shared ``ingest:domain:end`` topic and wants events for one tenant
    only. Composes with any callback-registry-bearing surface
    (the ingestion manager's ``lifecycle_callbacks``, a backend's
    ``state_write_callbacks``, an execution tracker's
    ``execution_callbacks``).

    Example::

        registry.register(
            INGEST_DOMAIN_END,
            TenantFilteredCallback(acme_handler, tenant_id="acme"),
        )

    The adapter is sync; for an async inner callback the wrapped
    callable returns the inner's awaitable, which the registry awaits
    under ``fire_async`` (the wrapper sees the opaque event mapping
    either way).
    """

    def __init__(
        self,
        inner: Callable[[Mapping[str, Any]], Any],
        *,
        tenant_id: str,
    ) -> None:
        self._inner = inner
        self._tenant_id = tenant_id

    def __call__(self, event: Mapping[str, Any]) -> Any:
        if event.get("tenant_id") != self._tenant_id:
            return None
        return self._inner(event)

    def __repr__(self) -> str:
        return (
            f"TenantFilteredCallback("
            f"inner={self._inner!r}, tenant_id={self._tenant_id!r})"
        )

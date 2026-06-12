"""Key-layout vocabulary for knowledge resource backends.

Names the three classes of keys every in-tree backend uses so external
event sources (S3 → EventBridge, filesystem inotify, GCS Pub/Sub) can
filter on the consumer-controlled subset and skip the DK-managed state
writes that would otherwise create a positive feedback loop during
ingest.

See :class:`KnowledgeKeyKind` for the enum members and
:meth:`KnowledgeResourceBackend.classify_key` /
:meth:`KnowledgeResourceBackend.key_pattern` for the consumer surface.
"""

from __future__ import annotations

from enum import Enum


class KnowledgeKeyKind(str, Enum):
    """The class of a key relative to the backend's layout.

    Members:
        CONTENT: consumer-controlled, written by ``put_file()``.
            External event triggers SHOULD subscribe to this kind.
        METADATA: DK-managed KB info + file index, written by
            ``_save_metadata()`` after every mutating operation and
            ingestion-status transition. External event triggers MUST
            NOT subscribe to this kind unless they are explicitly
            auditing state changes (rare).
        SNAPSHOT: DK-managed per-version content snapshot, written by
            ``_record_snapshot()`` once per content-changing mutation
            in snapshot mode. External event triggers MUST NOT
            subscribe to this kind unless they are explicitly
            archiving snapshot history.
        UNKNOWN: key did not match any declared layout segment. A
            future backend kind would surface as ``UNKNOWN`` on an
            older protocol consumer; a future protocol revision
            extending the enum keeps existing-consumer code valid.
    """

    CONTENT = "content"
    METADATA = "metadata"
    SNAPSHOT = "snapshot"
    UNKNOWN = "unknown"

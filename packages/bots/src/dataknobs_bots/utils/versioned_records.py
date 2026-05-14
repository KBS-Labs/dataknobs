"""Helpers for registries built on the dual-write versioning pattern.

Several registries in ``dataknobs_bots`` (``ArtifactRegistry``,
``RubricRegistry``) compose :class:`AsyncKeyedRecordStore` and use a
dual-write storage shape: every write produces a *latest pointer* row
keyed by the entity id and a *versioned snapshot* row keyed by
``f"{id}:{version}"``.  Both rows carry an identical
``data["_version_key"]`` of ``f"{id}:{version}"``.

When a query returns the raw record stream, callers must:

1. Skip versioned-snapshot rows so only latest pointers survive
   (snapshots and pointers share the same content for the current
   version, but old snapshots from prior versions would otherwise leak
   stale data).
2. Defensively deduplicate by entity id in case the backend ever
   returns the same pointer twice.

:func:`iter_latest_records` performs both passes in one place so the
two registries — and any future registry adopting the same pattern —
do not reimplement the dedup logic.

The helper is intentionally scoped to ``dataknobs-bots`` rather than
pushed into ``dataknobs-data``: the ``_version_key`` convention is a
bots-level layering on top of the generic
:class:`AsyncKeyedRecordStore`, not a property of the store itself.

Future extension:
    If a third package (e.g. ``dataknobs-fsm`` history versioning, or
    a new registry in ``dataknobs-llm``) adopts the same dual-write
    shape, the natural next step is to promote ``_version_key`` to a
    documented data-layer convention and move this helper into
    ``dataknobs-data`` — the binding contract is the
    ``data["_version_key"]`` field, not the bots-layer registries
    that currently use it.  Until then, keeping the helper here
    avoids leaking a bots-level idiom into the generic store API.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

from dataknobs_data import Record

logger = logging.getLogger(__name__)


def iter_latest_records(records: Iterable[Record]) -> Iterator[Record]:
    """Yield latest-pointer records from a versioned dual-write store.

    A record is considered a *versioned snapshot* when its storage key
    matches its ``data["_version_key"]`` — those are skipped.  Among
    the remaining latest-pointer records, only the first occurrence
    per ``data["id"]`` is yielded (defensive dedup against backends
    that might return a pointer more than once).  Records without an
    ``id`` field bypass the dedup table and are passed through.

    The first-occurrence rule preserves the input ordering, which
    matters when the caller has pushed a sort spec down to the
    backend: the underlying database's order is the order the helper
    yields.

    Args:
        records: Iterable of records from a versioned dual-write store
            (e.g., the result of ``AsyncKeyedRecordStore.search()``).

    Yields:
        Records that represent the latest pointer for each entity,
        in input order.
    """
    seen_ids: set[str] = set()
    for record in records:
        data = record.data or {}
        version_key = data.get("_version_key")
        record_key = record.storage_id or record.id
        if version_key and record_key == version_key:
            continue
        record_id = data.get("id")
        if record_id is None and not version_key:
            # Neither a recognizable pointer (has id) nor a versioned
            # snapshot (storage_id == _version_key).  Most likely a
            # legitimate record from a non-versioned store sharing the
            # database, but could also be corruption (pointer with the
            # entity id field stripped).  Log at debug so production
            # noise stays low while diagnosis is possible.
            logger.debug(
                "iter_latest_records: yielding record with no id "
                "and no _version_key (storage_id=%r)",
                record.storage_id,
            )
        if record_id is not None:
            if record_id in seen_ids:
                continue
            seen_ids.add(record_id)
        yield record

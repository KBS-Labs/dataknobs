# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""What the two S3 backends store, apart from how each reaches S3."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from dataknobs_common.clock import now_iso

from ..records import Record

if TYPE_CHECKING:
    from collections.abc import Mapping


class S3ObjectMixin:
    """The stored form of a record, and the two stamps a write gives it.

    No I/O: the sync and async twins share the rule and differ in transport.
    """

    @staticmethod
    def _stored_metadata(body: bytes) -> dict[str, Any]:
        """The metadata of a stored object, empty where it holds none.

        Raises:
            json.JSONDecodeError: The object is not JSON. A write over it
                would discard what is there, so it is not read as empty.
        """
        stored = json.loads(body)
        metadata = stored.get("metadata") if isinstance(stored, dict) else None
        return metadata if isinstance(metadata, dict) else {}

    def _object_for_write(
        self, record: Record, id: str, stored: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """The object that stores ``record`` under ``id``, stamped.

        ``created_at`` is the stored object's where it has one, else the
        caller's, so a copied record keeps its time, else now. ``updated_at``
        is now, from the same read of the clock. The caller's record is not
        written to: one passed to two writes would carry the first one's
        stamps and id into the second.

        Args:
            record: What to store.
            id: The id it is stored under, whatever id the record carries.
            stored: The metadata of the object being replaced, if there is one.
        """
        now = now_iso()
        metadata = dict(record.metadata or {})
        created_at = (stored or {}).get("created_at")
        if created_at is None:
            created_at = metadata.get("created_at")
        if isinstance(created_at, datetime):
            created_at = created_at.isoformat()
        metadata["id"] = id
        metadata["created_at"] = now if created_at is None else created_at
        metadata["updated_at"] = now
        return Record(data=record.fields, metadata=metadata, id=id).to_dict(
            include_metadata=True, flatten=False
        )

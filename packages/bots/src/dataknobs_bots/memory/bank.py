"""MemoryBank — typed collection of structured records for wizard data collection.

MemoryBank is a sibling to the existing ``Memory`` ABC (cross-conversation
recall), not a replacement.  It manages *lists* of structured records
(ingredients, contacts, configuration items) collected during a wizard flow.

Banks are managed by ``WizardReasoning`` and backed by ``SyncDatabase`` /
``AsyncDatabase`` from dataknobs-data.

Example:
    >>> from dataknobs_data.backends.memory import SyncMemoryDatabase
    >>> bank = MemoryBank("ingredients", schema={"required": ["name"]},
    ...                   db=SyncMemoryDatabase())
    >>> record_id = bank.add({"name": "flour", "amount": "2 cups"})
    >>> bank.count()
    1
    >>> bank.get(record_id).data["name"]
    'flour'
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from dataknobs_data import Record, SyncDatabase

logger = logging.getLogger(__name__)


@dataclass
class BankRecord:
    """Single record in a MemoryBank with provenance metadata."""

    record_id: str
    data: dict[str, Any]
    source_stage: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "record_id": self.record_id,
            "data": dict(self.data),
            "source_stage": self.source_stage,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BankRecord:
        """Deserialize from a plain dict."""
        return cls(
            record_id=d["record_id"],
            data=dict(d.get("data", {})),
            source_stage=d.get("source_stage", ""),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
        )


class MemoryBank:
    """Typed collection of structured records for wizard data collection.

    Backed by any ``SyncDatabase`` backend (memory, SQLite, PostgreSQL, etc.).

    Args:
        name: Bank identifier (e.g. ``"ingredients"``).
        schema: JSON-Schema-like dict.  Only ``required`` is enforced
            in Phase 1 (fields that must be present and non-None).
        db: Sync database backend for record storage.
        max_records: Optional cap on the number of records.
        duplicate_strategy: How to handle duplicates:
            ``"allow"`` (default), ``"reject"``, or ``"merge"``.
        match_fields: Fields used for duplicate detection.  ``None``
            means all data fields are compared.
        storage_mode: ``"inline"`` serialises all records in ``to_dict()``;
            ``"external"`` stores only the bank reference (records live
            in the persistent backend).
    """

    def __init__(
        self,
        name: str,
        schema: dict[str, Any],
        db: SyncDatabase,
        *,
        max_records: int | None = None,
        duplicate_strategy: str = "allow",
        match_fields: list[str] | None = None,
        storage_mode: str = "inline",
    ) -> None:
        self._name = name
        self._schema = schema
        self._db = db
        self._max_records = max_records
        self._duplicate_strategy = duplicate_strategy
        self._match_fields = match_fields
        self._storage_mode = storage_mode

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def name(self) -> str:
        """Bank identifier."""
        return self._name

    @property
    def schema(self) -> dict[str, Any]:
        """Schema definition for this bank's records."""
        return self._schema

    # -----------------------------------------------------------------
    # CRUD
    # -----------------------------------------------------------------

    def add(self, data: dict[str, Any], source_stage: str = "") -> str:
        """Add a record to the bank.

        Args:
            data: Field values for the new record.
            source_stage: Name of the wizard stage that produced this record.

        Returns:
            The generated record ID.

        Raises:
            ValueError: If required fields are missing, the bank is full,
                or duplicate_strategy is ``"reject"`` and a duplicate exists.
        """
        self._validate(data)

        if self._max_records is not None and self.count() >= self._max_records:
            raise ValueError(
                f"Bank '{self._name}' is full "
                f"(max_records={self._max_records})"
            )

        # Duplicate detection
        if self._duplicate_strategy != "allow":
            existing = self._check_duplicate(data)
            if existing is not None:
                if self._duplicate_strategy == "reject":
                    logger.debug(
                        "Duplicate rejected in bank '%s': matches record %s",
                        self._name,
                        existing.record_id,
                    )
                    return existing.record_id
                if self._duplicate_strategy == "merge":
                    merged = {**existing.data, **data}
                    self.update(existing.record_id, merged)
                    logger.debug(
                        "Duplicate merged in bank '%s': updated record %s",
                        self._name,
                        existing.record_id,
                    )
                    return existing.record_id

        now = time.time()
        record_id = uuid.uuid4().hex[:12]
        bank_record = BankRecord(
            record_id=record_id,
            data=dict(data),
            source_stage=source_stage,
            created_at=now,
            updated_at=now,
        )
        db_record = Record(
            data=bank_record.data,
            metadata={
                "record_id": record_id,
                "source_stage": source_stage,
                "created_at": now,
                "updated_at": now,
            },
        )
        self._db.create(db_record)
        logger.debug(
            "Added record %s to bank '%s' (count=%d)",
            record_id,
            self._name,
            self.count(),
        )
        return record_id

    def get(self, record_id: str) -> BankRecord | None:
        """Retrieve a record by ID.  Returns ``None`` if not found."""
        for bank_record in self.all():
            if bank_record.record_id == record_id:
                return bank_record
        return None

    def update(self, record_id: str, data: dict[str, Any]) -> bool:
        """Update a record's data fields.

        Returns:
            ``True`` if the record was found and updated.
        """
        all_records = self._db_records()
        for db_record in all_records:
            meta = db_record.metadata or {}
            if meta.get("record_id") == record_id:
                self._validate(data)
                now = time.time()
                updated_record = Record(
                    data=dict(data),
                    metadata={
                        **meta,
                        "updated_at": now,
                    },
                )
                self._db.update(db_record.storage_id, updated_record)
                logger.debug(
                    "Updated record %s in bank '%s'",
                    record_id,
                    self._name,
                )
                return True
        return False

    def remove(self, record_id: str) -> bool:
        """Remove a record by ID.

        Returns:
            ``True`` if the record was found and removed.
        """
        all_records = self._db_records()
        for db_record in all_records:
            meta = db_record.metadata or {}
            if meta.get("record_id") == record_id:
                self._db.delete(db_record.storage_id)
                logger.debug(
                    "Removed record %s from bank '%s'",
                    record_id,
                    self._name,
                )
                return True
        return False

    # -----------------------------------------------------------------
    # Collection operations
    # -----------------------------------------------------------------

    def count(self) -> int:
        """Number of records in the bank."""
        return len(self._db_records())

    def all(self) -> list[BankRecord]:
        """Return all records, ordered by creation time."""
        records = [self._to_bank_record(r) for r in self._db_records()]
        records.sort(key=lambda r: r.created_at)
        return records

    def clear(self) -> None:
        """Remove all records from the bank."""
        for db_record in self._db_records():
            self._db.delete(db_record.storage_id)
        logger.debug("Cleared bank '%s'", self._name)

    # -----------------------------------------------------------------
    # Search / filter (Phase 3)
    # -----------------------------------------------------------------

    def find(self, **field_values: Any) -> list[BankRecord]:
        """Find records matching exact field values.

        Args:
            **field_values: Field name/value pairs to match.

        Returns:
            List of matching ``BankRecord`` objects.
        """
        results: list[BankRecord] = []
        for bank_record in self.all():
            if all(
                bank_record.data.get(k) == v for k, v in field_values.items()
            ):
                results.append(bank_record)
        return results

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def _validate(self, data: dict[str, Any]) -> None:
        """Check required fields are present and non-None.

        Raises:
            ValueError: If any required field is missing or ``None``.
        """
        required = self._schema.get("required", [])
        missing = [f for f in required if data.get(f) is None]
        if missing:
            raise ValueError(
                f"Bank '{self._name}': missing required fields: {missing}"
            )

    # -----------------------------------------------------------------
    # Duplicate detection (Phase 3)
    # -----------------------------------------------------------------

    def _check_duplicate(self, data: dict[str, Any]) -> BankRecord | None:
        """Check if a record with matching fields already exists.

        Uses ``match_fields`` if set, otherwise compares all data fields.

        Returns:
            The existing ``BankRecord`` if a duplicate is found,
            ``None`` otherwise.
        """
        fields_to_check = self._match_fields or list(data.keys())
        for existing in self.all():
            if all(
                existing.data.get(f) == data.get(f) for f in fields_to_check
            ):
                return existing
        return None

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _db_records(self) -> list[Record]:
        """Retrieve all raw ``Record`` objects from the backing database."""
        from dataknobs_data import Query

        result = self._db.search(Query())
        return list(result) if result else []

    @staticmethod
    def _to_bank_record(db_record: Record) -> BankRecord:
        """Convert a dataknobs ``Record`` to a ``BankRecord``."""
        meta = db_record.metadata or {}
        return BankRecord(
            record_id=meta.get("record_id", ""),
            data=dict(db_record.data) if db_record.data else {},
            source_stage=meta.get("source_stage", ""),
            created_at=meta.get("created_at", 0.0),
            updated_at=meta.get("updated_at", 0.0),
        )

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        self._db.close()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the bank to a plain dict.

        In ``inline`` mode, all records are included.
        In ``external`` mode, only the bank reference is stored.
        """
        base: dict[str, Any] = {
            "name": self._name,
            "schema": self._schema,
            "max_records": self._max_records,
            "duplicate_strategy": self._duplicate_strategy,
            "match_fields": self._match_fields,
            "storage_mode": self._storage_mode,
        }
        if self._storage_mode == "inline":
            base["records"] = [r.to_dict() for r in self.all()]
        return base

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], db: SyncDatabase | None = None
    ) -> MemoryBank:
        """Deserialize a bank from a plain dict.

        Args:
            d: Serialized bank dict (from ``to_dict()``).
            db: Optional pre-configured database backend.  When ``None``
                a fresh ``SyncMemoryDatabase`` is created and any
                serialized records are re-inserted (inline mode).
        """
        if db is None:
            from dataknobs_data.backends.memory import SyncMemoryDatabase

            db = SyncMemoryDatabase()
        bank = cls(
            name=d["name"],
            schema=d.get("schema", {}),
            db=db,
            max_records=d.get("max_records"),
            duplicate_strategy=d.get("duplicate_strategy", "allow"),
            match_fields=d.get("match_fields"),
            storage_mode=d.get("storage_mode", "inline"),
        )
        # Re-insert persisted records (present only in inline mode)
        for rec_dict in d.get("records", []):
            bank_record = BankRecord.from_dict(rec_dict)
            db_record = Record(
                data=bank_record.data,
                metadata={
                    "record_id": bank_record.record_id,
                    "source_stage": bank_record.source_stage,
                    "created_at": bank_record.created_at,
                    "updated_at": bank_record.updated_at,
                },
            )
            db.create(db_record)
        return bank


class EmptyBankProxy:
    """Null-object returned by ``bank('nonexistent')``.

    Prevents crashes in conditions and templates when a bank name is
    referenced but not configured.  Every method returns a safe default.
    """

    def __init__(self, name: str = "") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def schema(self) -> dict[str, Any]:
        return {}

    def add(self, data: dict[str, Any], source_stage: str = "") -> str:
        logger.warning("add() called on EmptyBankProxy '%s'", self._name)
        return ""

    def get(self, record_id: str) -> None:
        return None

    def update(self, record_id: str, data: dict[str, Any]) -> bool:
        return False

    def remove(self, record_id: str) -> bool:
        return False

    def count(self) -> int:
        return 0

    def all(self) -> list[BankRecord]:
        return []

    def clear(self) -> None:
        pass

    def find(self, **field_values: Any) -> list[BankRecord]:
        return []


class AsyncMemoryBank:
    """Async MemoryBank backed by any ``AsyncDatabase`` backend.

    Provides the same interface as ``MemoryBank`` but with async methods
    for use with external storage backends (SQLite, PostgreSQL, etc.).

    Args:
        name: Bank identifier.
        schema: JSON-Schema-like dict (``required`` enforced in Phase 1).
        db: Async database backend for record storage.
        max_records: Optional cap on the number of records.
        duplicate_strategy: ``"allow"`` | ``"reject"`` | ``"merge"``.
        match_fields: Fields used for duplicate detection.
        storage_mode: ``"inline"`` serialises all records;
            ``"external"`` stores only a bank reference.
    """

    def __init__(
        self,
        name: str,
        schema: dict[str, Any],
        db: Any,  # AsyncDatabase
        *,
        max_records: int | None = None,
        duplicate_strategy: str = "allow",
        match_fields: list[str] | None = None,
        storage_mode: str = "inline",
    ) -> None:
        self._name = name
        self._schema = schema
        self._db = db
        self._max_records = max_records
        self._duplicate_strategy = duplicate_strategy
        self._match_fields = match_fields
        self._storage_mode = storage_mode

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def schema(self) -> dict[str, Any]:
        return self._schema

    # -----------------------------------------------------------------
    # CRUD
    # -----------------------------------------------------------------

    async def add(self, data: dict[str, Any], source_stage: str = "") -> str:
        """Add a record to the bank."""
        self._validate(data)

        if self._max_records is not None:
            current = await self.count()
            if current >= self._max_records:
                raise ValueError(
                    f"Bank '{self._name}' is full "
                    f"(max_records={self._max_records})"
                )

        # Duplicate detection
        if self._duplicate_strategy != "allow":
            existing = await self._check_duplicate(data)
            if existing is not None:
                if self._duplicate_strategy == "reject":
                    return existing.record_id
                if self._duplicate_strategy == "merge":
                    merged = {**existing.data, **data}
                    await self.update(existing.record_id, merged)
                    return existing.record_id

        now = time.time()
        record_id = uuid.uuid4().hex[:12]
        db_record = Record(
            data=dict(data),
            metadata={
                "record_id": record_id,
                "source_stage": source_stage,
                "created_at": now,
                "updated_at": now,
            },
        )
        await self._db.create(db_record)
        logger.debug(
            "Added record %s to async bank '%s'",
            record_id,
            self._name,
        )
        return record_id

    async def get(self, record_id: str) -> BankRecord | None:
        """Retrieve a record by ID."""
        for bank_record in await self.all():
            if bank_record.record_id == record_id:
                return bank_record
        return None

    async def update(self, record_id: str, data: dict[str, Any]) -> bool:
        """Update a record's data fields."""
        for db_record in await self._db_records():
            meta = db_record.metadata or {}
            if meta.get("record_id") == record_id:
                self._validate(data)
                now = time.time()
                updated_record = Record(
                    data=dict(data),
                    metadata={**meta, "updated_at": now},
                )
                await self._db.update(db_record.storage_id, updated_record)
                return True
        return False

    async def remove(self, record_id: str) -> bool:
        """Remove a record by ID."""
        for db_record in await self._db_records():
            meta = db_record.metadata or {}
            if meta.get("record_id") == record_id:
                await self._db.delete(db_record.storage_id)
                return True
        return False

    # -----------------------------------------------------------------
    # Collection operations
    # -----------------------------------------------------------------

    async def count(self) -> int:
        return len(await self._db_records())

    async def all(self) -> list[BankRecord]:
        records = [
            MemoryBank._to_bank_record(r) for r in await self._db_records()
        ]
        records.sort(key=lambda r: r.created_at)
        return records

    async def clear(self) -> None:
        for db_record in await self._db_records():
            await self._db.delete(db_record.storage_id)

    async def find(self, **field_values: Any) -> list[BankRecord]:
        results: list[BankRecord] = []
        for bank_record in await self.all():
            if all(
                bank_record.data.get(k) == v
                for k, v in field_values.items()
            ):
                results.append(bank_record)
        return results

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def _validate(self, data: dict[str, Any]) -> None:
        required = self._schema.get("required", [])
        missing = [f for f in required if data.get(f) is None]
        if missing:
            raise ValueError(
                f"Bank '{self._name}': missing required fields: {missing}"
            )

    # -----------------------------------------------------------------
    # Duplicate detection
    # -----------------------------------------------------------------

    async def _check_duplicate(
        self, data: dict[str, Any]
    ) -> BankRecord | None:
        fields_to_check = self._match_fields or list(data.keys())
        for existing in await self.all():
            if all(
                existing.data.get(f) == data.get(f)
                for f in fields_to_check
            ):
                return existing
        return None

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    async def _db_records(self) -> list[Record]:
        from dataknobs_data import Query

        result = await self._db.search(Query())
        return list(result) if result else []

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    async def to_dict(self) -> dict[str, Any]:
        """Serialize the bank to a plain dict.

        In ``inline`` mode, all records are included.
        In ``external`` mode, only the bank reference is stored.
        """
        base: dict[str, Any] = {
            "name": self._name,
            "schema": self._schema,
            "max_records": self._max_records,
            "duplicate_strategy": self._duplicate_strategy,
            "match_fields": self._match_fields,
            "storage_mode": self._storage_mode,
        }
        if self._storage_mode == "inline":
            base["records"] = [r.to_dict() for r in await self.all()]
        return base

    @classmethod
    async def from_dict(cls, d: dict[str, Any]) -> AsyncMemoryBank:
        """Deserialize a bank from a plain dict."""
        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        db = AsyncMemoryDatabase()
        bank = cls(
            name=d["name"],
            schema=d.get("schema", {}),
            db=db,
            max_records=d.get("max_records"),
            duplicate_strategy=d.get("duplicate_strategy", "allow"),
            match_fields=d.get("match_fields"),
            storage_mode=d.get("storage_mode", "inline"),
        )
        for rec_dict in d.get("records", []):
            bank_record = BankRecord.from_dict(rec_dict)
            db_record = Record(
                data=bank_record.data,
                metadata={
                    "record_id": bank_record.record_id,
                    "source_stage": bank_record.source_stage,
                    "created_at": bank_record.created_at,
                    "updated_at": bank_record.updated_at,
                },
            )
            await db.create(db_record)
        return bank

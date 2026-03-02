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
from typing import Any, Protocol, runtime_checkable

from dataknobs_data import Record, SyncDatabase

logger = logging.getLogger(__name__)


@dataclass
class BankRecord:
    """Single record in a MemoryBank with provenance metadata.

    Attributes:
        record_id: Unique identifier (12-char hex).
        data: Field values for this record.
        source_stage: Wizard stage that originally created this record.
        created_at: Unix timestamp of creation.
        updated_at: Unix timestamp of last modification.
        modified_in_stage: Wizard stage that last modified this record.
            Empty string means never modified after creation.
    """

    record_id: str
    data: dict[str, Any]
    source_stage: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    modified_in_stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "record_id": self.record_id,
            "data": dict(self.data),
            "source_stage": self.source_stage,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "modified_in_stage": self.modified_in_stage,
        }

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], *, strict: bool = True
    ) -> BankRecord:
        """Deserialize from a plain dict.

        Args:
            d: Serialized record dict (from ``to_dict()``).
            strict: When ``True`` (default), raise ``ValueError`` if
                ``created_at`` or ``updated_at`` is missing.  When
                ``False``, fall back to ``time.time()`` for missing
                timestamps (legacy data migration).
        """
        if strict:
            missing = [
                f for f in ("created_at", "updated_at") if f not in d
            ]
            if missing:
                raise ValueError(
                    f"BankRecord.from_dict(): missing required timestamp "
                    f"fields: {missing}"
                )
        return cls(
            record_id=d["record_id"],
            data=dict(d.get("data", {})),
            source_stage=d.get("source_stage", ""),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            modified_in_stage=d.get("modified_in_stage"),
        )


# =====================================================================
# Protocols
# =====================================================================


@runtime_checkable
class SyncBankProtocol(Protocol):
    """Protocol for synchronous memory bank implementations.

    Defines the public interface shared by ``MemoryBank`` and
    ``EmptyBankProxy``.  Use this for type annotations when the
    concrete class doesn't matter.

    Implementations:
        - ``MemoryBank`` — full implementation backed by ``SyncDatabase``
        - ``EmptyBankProxy`` — null-object returning safe defaults
    """

    @property
    def name(self) -> str: ...

    @property
    def schema(self) -> dict[str, Any]: ...

    @property
    def match_fields(self) -> list[str] | None: ...

    def add(self, data: dict[str, Any], source_stage: str = "") -> str: ...

    def get(self, record_id: str) -> BankRecord | None: ...

    def update(
        self,
        record_id: str,
        data: dict[str, Any],
        modified_in_stage: str = "",
    ) -> bool: ...

    def remove(self, record_id: str) -> bool: ...

    def count(self) -> int: ...

    def all(self) -> list[BankRecord]: ...

    def clear(self) -> None: ...

    def find(self, **field_values: Any) -> list[BankRecord]: ...

    def to_dict(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


@runtime_checkable
class AsyncBankProtocol(Protocol):
    """Protocol for asynchronous memory bank implementations.

    Defines the public interface for ``AsyncMemoryBank``.

    Implementations:
        - ``AsyncMemoryBank`` — full implementation backed by
          ``AsyncDatabase``
    """

    @property
    def name(self) -> str: ...

    @property
    def schema(self) -> dict[str, Any]: ...

    @property
    def match_fields(self) -> list[str] | None: ...

    async def add(
        self, data: dict[str, Any], source_stage: str = ""
    ) -> str: ...

    async def get(self, record_id: str) -> BankRecord | None: ...

    async def update(
        self,
        record_id: str,
        data: dict[str, Any],
        modified_in_stage: str = "",
    ) -> bool: ...

    async def remove(self, record_id: str) -> bool: ...

    async def count(self) -> int: ...

    async def all(self) -> list[BankRecord]: ...

    async def clear(self) -> None: ...

    async def find(self, **field_values: Any) -> list[BankRecord]: ...

    async def to_dict(self) -> dict[str, Any]: ...


# =====================================================================
# Shared core logic
# =====================================================================


class _BankCore:
    """Shared logic for MemoryBank and AsyncMemoryBank.

    Holds configuration state and provides pure-logic methods
    (validation, record creation, duplicate checking, serialization
    config).  Does NOT call the database — callers handle all DB I/O.
    """

    def __init__(
        self,
        name: str,
        schema: dict[str, Any],
        *,
        max_records: int | None = None,
        duplicate_strategy: str = "allow",
        match_fields: list[str] | None = None,
        storage_mode: str = "inline",
    ) -> None:
        self._name = name
        self._schema = schema
        self._max_records = max_records
        self._duplicate_strategy = duplicate_strategy
        self._match_fields = match_fields
        self._storage_mode = storage_mode

    # -- Properties --

    @property
    def name(self) -> str:
        """Bank identifier."""
        return self._name

    @property
    def schema(self) -> dict[str, Any]:
        """Schema definition for this bank's records."""
        return self._schema

    @property
    def match_fields(self) -> list[str] | None:
        """Fields used for duplicate detection."""
        return self._match_fields

    @property
    def max_records(self) -> int | None:
        """Optional cap on the number of records."""
        return self._max_records

    @property
    def duplicate_strategy(self) -> str:
        """How to handle duplicates: ``allow``, ``reject``, or ``merge``."""
        return self._duplicate_strategy

    @property
    def storage_mode(self) -> str:
        """Serialization mode: ``inline`` or ``external``."""
        return self._storage_mode

    # -- Validation (pure) --

    def validate(self, data: dict[str, Any]) -> None:
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

    # -- Capacity check (pure) --

    def check_capacity(self, current_count: int) -> None:
        """Raise ``ValueError`` if adding would exceed ``max_records``.

        Args:
            current_count: Current number of records in the bank.
        """
        if (
            self._max_records is not None
            and current_count >= self._max_records
        ):
            raise ValueError(
                f"Bank '{self._name}' is full "
                f"(max_records={self._max_records})"
            )

    # -- Duplicate detection (pure — caller passes existing records) --

    def check_duplicate(
        self,
        data: dict[str, Any],
        existing_records: list[BankRecord],
    ) -> BankRecord | None:
        """Check if a record with matching fields already exists.

        Uses ``match_fields`` if set, otherwise compares the union of
        keys from both the new and existing records.

        Args:
            data: New record data.
            existing_records: All current ``BankRecord`` objects.

        Returns:
            The existing ``BankRecord`` if a duplicate is found,
            ``None`` otherwise.
        """
        for existing in existing_records:
            fields_to_check = self._match_fields or list(
                set(data.keys()) | set(existing.data.keys())
            )
            if all(
                existing.data.get(f) == data.get(f) for f in fields_to_check
            ):
                return existing
        return None

    # -- Record creation (pure) --

    def create_bank_record(
        self,
        data: dict[str, Any],
        source_stage: str = "",
    ) -> tuple[BankRecord, Record]:
        """Create a ``BankRecord`` and its corresponding DB ``Record``.

        Generates a unique record ID and current timestamps.

        Returns:
            Tuple of ``(BankRecord, Record)`` ready for DB insertion.
        """
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
            storage_id=record_id,
        )
        return bank_record, db_record

    def create_updated_record(
        self,
        data: dict[str, Any],
        existing_meta: dict[str, Any],
        modified_in_stage: str = "",
    ) -> Record:
        """Create an updated DB ``Record`` from new data and existing metadata.

        Args:
            data: New field values.
            existing_meta: Existing DB record metadata.
            modified_in_stage: Wizard stage performing the update.

        Returns:
            ``Record`` ready for DB update.
        """
        now = time.time()
        updated_meta = {**existing_meta, "updated_at": now}
        if modified_in_stage:
            updated_meta["modified_in_stage"] = modified_in_stage
        record = Record(data=dict(data), metadata=updated_meta)
        rid = existing_meta.get("record_id", "")
        if rid:
            record.storage_id = rid
        return record

    # -- DB Record ↔ BankRecord conversion (static, pure) --

    @staticmethod
    def to_bank_record(db_record: Record) -> BankRecord:
        """Convert a dataknobs ``Record`` to a ``BankRecord``."""
        meta = db_record.metadata or {}
        return BankRecord(
            record_id=meta.get("record_id", ""),
            data=dict(db_record.data) if db_record.data else {},
            source_stage=meta.get("source_stage", ""),
            created_at=meta.get("created_at", 0.0),
            updated_at=meta.get("updated_at", 0.0),
            modified_in_stage=meta.get("modified_in_stage"),
        )

    @staticmethod
    def bank_record_to_db_record(bank_record: BankRecord) -> Record:
        """Convert a ``BankRecord`` back to a DB ``Record``.

        Used during ``from_dict()`` deserialization.
        """
        return Record(
            data=bank_record.data,
            metadata={
                "record_id": bank_record.record_id,
                "source_stage": bank_record.source_stage,
                "created_at": bank_record.created_at,
                "updated_at": bank_record.updated_at,
                "modified_in_stage": bank_record.modified_in_stage,
            },
            storage_id=bank_record.record_id,
        )

    # -- Serialization helpers (pure) --

    def to_config_dict(self) -> dict[str, Any]:
        """Return the bank configuration as a dict (no records)."""
        return {
            "name": self._name,
            "schema": self._schema,
            "max_records": self._max_records,
            "duplicate_strategy": self._duplicate_strategy,
            "match_fields": self._match_fields,
            "storage_mode": self._storage_mode,
        }

    @staticmethod
    def extract_config(d: dict[str, Any]) -> dict[str, Any]:
        """Extract constructor kwargs from a serialized dict."""
        return {
            "name": d["name"],
            "schema": d.get("schema", {}),
            "max_records": d.get("max_records"),
            "duplicate_strategy": d.get("duplicate_strategy", "allow"),
            "match_fields": d.get("match_fields"),
            "storage_mode": d.get("storage_mode", "inline"),
        }


# =====================================================================
# Sync implementation
# =====================================================================


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
        self._core = _BankCore(
            name=name,
            schema=schema,
            max_records=max_records,
            duplicate_strategy=duplicate_strategy,
            match_fields=match_fields,
            storage_mode=storage_mode,
        )
        self._db = db

    # -----------------------------------------------------------------
    # Properties (delegate to core)
    # -----------------------------------------------------------------

    @property
    def name(self) -> str:
        """Bank identifier."""
        return self._core.name

    @property
    def schema(self) -> dict[str, Any]:
        """Schema definition for this bank's records."""
        return self._core.schema

    @property
    def match_fields(self) -> list[str] | None:
        """Fields used for duplicate detection, or ``None`` for all-field comparison."""
        return self._core.match_fields

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
        self._core.validate(data)
        self._core.check_capacity(self.count())

        # Duplicate detection
        if self._core.duplicate_strategy != "allow":
            existing = self._core.check_duplicate(data, self.all())
            if existing is not None:
                if self._core.duplicate_strategy == "reject":
                    logger.debug(
                        "Duplicate rejected in bank '%s': matches record %s",
                        self._core.name,
                        existing.record_id,
                    )
                    return existing.record_id
                if self._core.duplicate_strategy == "merge":
                    merged = {**existing.data, **data}
                    self.update(existing.record_id, merged)
                    logger.debug(
                        "Duplicate merged in bank '%s': updated record %s",
                        self._core.name,
                        existing.record_id,
                    )
                    return existing.record_id

        bank_record, db_record = self._core.create_bank_record(
            data, source_stage
        )
        self._db.create(db_record)
        logger.debug(
            "Added record %s to bank '%s' (count=%d)",
            bank_record.record_id,
            self._core.name,
            self.count(),
        )
        return bank_record.record_id

    def get(self, record_id: str) -> BankRecord | None:
        """Retrieve a record by ID.  Returns ``None`` if not found."""
        db_record = self._db.read(record_id)
        if db_record is None:
            return None
        return _BankCore.to_bank_record(db_record)

    def update(
        self,
        record_id: str,
        data: dict[str, Any],
        modified_in_stage: str = "",
    ) -> bool:
        """Update a record's data fields.

        Args:
            record_id: ID of the record to update.
            data: New field values.
            modified_in_stage: Wizard stage performing the update
                (for provenance tracking).

        Returns:
            ``True`` if the record was found and updated.
        """
        db_record = self._db.read(record_id)
        if db_record is None:
            return False
        meta = db_record.metadata or {}
        self._core.validate(data)
        updated_record = self._core.create_updated_record(
            data, meta, modified_in_stage
        )
        self._db.update(record_id, updated_record)
        logger.debug(
            "Updated record %s in bank '%s'",
            record_id,
            self._core.name,
        )
        return True

    def remove(self, record_id: str) -> bool:
        """Remove a record by ID.

        Returns:
            ``True`` if the record was found and removed.
        """
        result = self._db.delete(record_id)
        if result:
            logger.debug(
                "Removed record %s from bank '%s'",
                record_id,
                self._core.name,
            )
        return result

    # -----------------------------------------------------------------
    # Collection operations
    # -----------------------------------------------------------------

    def count(self) -> int:
        """Number of records in the bank."""
        return len(self._db_records())

    def all(self) -> list[BankRecord]:
        """Return all records, ordered by creation time."""
        records = [
            _BankCore.to_bank_record(r) for r in self._db_records()
        ]
        records.sort(key=lambda r: r.created_at)
        return records

    def clear(self) -> None:
        """Remove all records from the bank."""
        for db_record in self._db_records():
            self._db.delete(db_record.storage_id)
        logger.debug("Cleared bank '%s'", self._core.name)

    # -----------------------------------------------------------------
    # Search / filter
    # -----------------------------------------------------------------

    def find(self, **field_values: Any) -> list[BankRecord]:
        """Find records matching exact field values.

        Args:
            **field_values: Field name/value pairs to match.

        Returns:
            List of matching ``BankRecord`` objects.
        """
        if not field_values:
            return self.all()
        from dataknobs_data import Filter, Operator, Query

        query = Query(
            filters=[
                Filter(k, Operator.EQ, v) for k, v in field_values.items()
            ]
        )
        results = self._db.search(query)
        records = [_BankCore.to_bank_record(r) for r in (results or [])]
        records.sort(key=lambda r: r.created_at)
        return records

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _db_records(self) -> list[Record]:
        """Retrieve all raw ``Record`` objects from the backing database."""
        from dataknobs_data import Query

        result = self._db.search(Query())
        return list(result) if result else []

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
        base = self._core.to_config_dict()
        if self._core.storage_mode == "inline":
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
        config = _BankCore.extract_config(d)
        bank = cls(db=db, **config)
        for rec_dict in d.get("records", []):
            bank_record = BankRecord.from_dict(rec_dict)
            db.create(_BankCore.bank_record_to_db_record(bank_record))
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

    @property
    def match_fields(self) -> list[str] | None:
        return None

    def add(self, data: dict[str, Any], source_stage: str = "") -> str:
        logger.warning("add() called on EmptyBankProxy '%s'", self._name)
        return ""

    def get(self, record_id: str) -> None:
        return None

    def update(
        self,
        record_id: str,
        data: dict[str, Any],
        modified_in_stage: str = "",
    ) -> bool:
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "schema": {},
            "max_records": None,
            "duplicate_strategy": "allow",
            "match_fields": None,
            "storage_mode": "inline",
            "records": [],
        }

    def close(self) -> None:
        pass


# =====================================================================
# Async implementation
# =====================================================================


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
        self._core = _BankCore(
            name=name,
            schema=schema,
            max_records=max_records,
            duplicate_strategy=duplicate_strategy,
            match_fields=match_fields,
            storage_mode=storage_mode,
        )
        self._db = db

    # -----------------------------------------------------------------
    # Properties (delegate to core)
    # -----------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._core.name

    @property
    def schema(self) -> dict[str, Any]:
        return self._core.schema

    @property
    def match_fields(self) -> list[str] | None:
        """Fields used for duplicate detection, or ``None`` for all-field comparison."""
        return self._core.match_fields

    # -----------------------------------------------------------------
    # CRUD
    # -----------------------------------------------------------------

    async def add(self, data: dict[str, Any], source_stage: str = "") -> str:
        """Add a record to the bank."""
        self._core.validate(data)
        self._core.check_capacity(await self.count())

        # Duplicate detection
        if self._core.duplicate_strategy != "allow":
            existing = self._core.check_duplicate(data, await self.all())
            if existing is not None:
                if self._core.duplicate_strategy == "reject":
                    return existing.record_id
                if self._core.duplicate_strategy == "merge":
                    merged = {**existing.data, **data}
                    await self.update(existing.record_id, merged)
                    return existing.record_id

        bank_record, db_record = self._core.create_bank_record(
            data, source_stage
        )
        await self._db.create(db_record)
        logger.debug(
            "Added record %s to async bank '%s'",
            bank_record.record_id,
            self._core.name,
        )
        return bank_record.record_id

    async def get(self, record_id: str) -> BankRecord | None:
        """Retrieve a record by ID."""
        db_record = await self._db.read(record_id)
        if db_record is None:
            return None
        return _BankCore.to_bank_record(db_record)

    async def update(
        self,
        record_id: str,
        data: dict[str, Any],
        modified_in_stage: str = "",
    ) -> bool:
        """Update a record's data fields.

        Args:
            record_id: ID of the record to update.
            data: New field values.
            modified_in_stage: Wizard stage performing the update
                (for provenance tracking).
        """
        db_record = await self._db.read(record_id)
        if db_record is None:
            return False
        meta = db_record.metadata or {}
        self._core.validate(data)
        updated_record = self._core.create_updated_record(
            data, meta, modified_in_stage
        )
        await self._db.update(record_id, updated_record)
        return True

    async def remove(self, record_id: str) -> bool:
        """Remove a record by ID."""
        result = await self._db.delete(record_id)
        return result

    # -----------------------------------------------------------------
    # Collection operations
    # -----------------------------------------------------------------

    async def count(self) -> int:
        return len(await self._db_records())

    async def all(self) -> list[BankRecord]:
        records = [
            _BankCore.to_bank_record(r) for r in await self._db_records()
        ]
        records.sort(key=lambda r: r.created_at)
        return records

    async def clear(self) -> None:
        for db_record in await self._db_records():
            await self._db.delete(db_record.storage_id)

    async def find(self, **field_values: Any) -> list[BankRecord]:
        if not field_values:
            return await self.all()
        from dataknobs_data import Filter, Operator, Query

        query = Query(
            filters=[
                Filter(k, Operator.EQ, v) for k, v in field_values.items()
            ]
        )
        results = await self._db.search(query)
        records = [_BankCore.to_bank_record(r) for r in (results or [])]
        records.sort(key=lambda r: r.created_at)
        return records

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
        base = self._core.to_config_dict()
        if self._core.storage_mode == "inline":
            base["records"] = [r.to_dict() for r in await self.all()]
        return base

    @classmethod
    async def from_dict(cls, d: dict[str, Any]) -> AsyncMemoryBank:
        """Deserialize a bank from a plain dict."""
        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        db = AsyncMemoryDatabase()
        config = _BankCore.extract_config(d)
        bank = cls(db=db, **config)
        for rec_dict in d.get("records", []):
            bank_record = BankRecord.from_dict(rec_dict)
            await db.create(_BankCore.bank_record_to_db_record(bank_record))
        return bank

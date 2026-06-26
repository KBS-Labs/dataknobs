"""Buffered, atomic transaction handle for :class:`AsyncDatabase`.

A :class:`BufferedTransaction` defers every write until :meth:`commit`. This
gives two guarantees:

1. **Universal rollback.** Because writes are buffered, raising before commit
   (e.g. from business logic inside an ``async with db.transaction()`` block)
   persists nothing — on *any* backend, transactional or not.
2. **Atomic commit on transactional backends.** The commit flush replays the
   buffer through the database's atomic batch primitives (``create_batch`` /
   ``delete_batch``), coalescing consecutive same-kind operations into a single
   batch call. On backends whose batch operations are wrapped in a backend
   transaction (sqlite/postgres/duckdb — those reporting
   :meth:`AsyncDatabase.supports_transactions`), each coalesced batch is
   all-or-nothing.

What it deliberately does **not** provide: in-transaction isolation or
read-your-writes. Buffered writes are invisible to reads (``db.read``) until
commit, and concurrent readers never see a partially-applied transaction
because nothing is written until the flush. Consumers needing connection-scoped
isolation should branch on :meth:`AsyncDatabase.supports_transactions` and use a
backend-native transaction directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dataknobs_common.exceptions import OperationError

if TYPE_CHECKING:
    from .database import AsyncDatabase
    from .records import Record

logger = logging.getLogger(__name__)

#: Accepted ``policy`` values for :meth:`AsyncDatabase.transaction` /
#: :meth:`AsyncDatabase.begin_transaction`.
VALID_TRANSACTION_POLICIES = ("strict", "emulate")


class BufferedTransaction:
    """A staged set of writes flushed atomically on :meth:`commit`.

    Obtain one via :meth:`AsyncDatabase.transaction` (context-manager form) or
    :meth:`AsyncDatabase.begin_transaction` (explicit ``commit``/``rollback``).
    The write methods (:meth:`create`, :meth:`create_batch`, :meth:`upsert`,
    :meth:`delete`) stage operations; nothing reaches the backend until
    :meth:`commit`.
    """

    def __init__(self, db: AsyncDatabase, *, policy: str = "strict") -> None:
        """Initialize a buffered transaction.

        Args:
            db: The owning async database.
            policy: ``"strict"`` or ``"emulate"`` — surfaced for introspection
                (the gating happens in :meth:`AsyncDatabase.begin_transaction`).
        """
        self._db = db
        self._policy = policy
        self._ops: list[tuple[str, Any]] = []
        self._closed = False
        # Capture the atomicity guarantee at begin-time so a consumer can branch
        # on it (and so the value is stable for the life of the handle).
        self._atomic = bool(db.supports_transactions())

    @property
    def is_atomic(self) -> bool:
        """Whether :meth:`commit` is all-or-nothing on the backing backend."""
        return self._atomic

    @property
    def policy(self) -> str:
        """The unsupported-backend policy this transaction was opened with."""
        return self._policy

    @property
    def closed(self) -> bool:
        """Whether the transaction has been committed or rolled back."""
        return self._closed

    def _stage(self, op: tuple[str, Any]) -> None:
        if self._closed:
            raise OperationError(
                "Cannot stage a write on a transaction that has already been "
                "committed or rolled back"
            )
        self._ops.append(op)

    async def create(self, record: Record) -> None:
        """Stage a record creation (flushed via ``create_batch`` on commit)."""
        self._stage(("create", record))

    async def create_batch(self, records: list[Record]) -> None:
        """Stage multiple record creations."""
        for record in records:
            self._stage(("create", record))

    async def upsert(
        self, id_or_record: str | Record, record: Record | None = None
    ) -> None:
        """Stage an upsert (same calling convention as ``AsyncDatabase.upsert``)."""
        self._stage(("upsert", (id_or_record, record)))

    async def delete(self, id: str) -> None:
        """Stage a delete (flushed via ``delete_batch`` on commit)."""
        self._stage(("delete", id))

    async def commit(self) -> dict[str, int]:
        """Flush all staged writes to the backend.

        Consecutive same-kind operations are coalesced into a single atomic
        batch call (``create_batch`` / ``delete_batch``); upserts are applied
        individually (the abstraction has no batch-upsert primitive). On a
        backend reporting :meth:`AsyncDatabase.supports_transactions`, each
        coalesced batch is all-or-nothing.

        Idempotent: a second call (after commit or rollback) is a no-op.

        Returns:
            ``{"affected_rows": <count>}``.
        """
        if self._closed:
            return {"affected_rows": 0}
        self._closed = True

        affected = 0
        ops = self._ops
        i = 0
        n = len(ops)
        while i < n:
            kind = ops[i][0]
            if kind == "create":
                records = []
                j = i
                while j < n and ops[j][0] == "create":
                    records.append(ops[j][1])
                    j += 1
                created = await self._db.create_batch(records)
                affected += len(created)
                i = j
            elif kind == "delete":
                ids = []
                j = i
                while j < n and ops[j][0] == "delete":
                    ids.append(ops[j][1])
                    j += 1
                results = await self._db.delete_batch(ids)
                affected += sum(1 for ok in results if ok)
                i = j
            else:  # "upsert"
                id_or_record, record = ops[i][1]
                if record is not None:
                    await self._db.upsert(id_or_record, record)
                else:
                    await self._db.upsert(id_or_record)
                affected += 1
                i += 1
        return {"affected_rows": affected}

    async def rollback(self) -> None:
        """Discard all staged writes. Idempotent."""
        self._closed = True
        self._ops.clear()

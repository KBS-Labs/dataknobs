"""Buffered transaction handle for :class:`AsyncDatabase`.

A :class:`BufferedTransaction` defers every write until :meth:`commit`. This
gives two guarantees:

1. **Universal rollback.** Because writes are buffered, raising before commit
   (e.g. from business logic inside an ``async with db.transaction()`` block)
   persists nothing — on *any* backend, transactional or not.
2. **Atomic commit of a single same-kind batch on transactional backends.** The
   commit flush replays the buffer through the database's atomic batch
   primitives (``create_batch`` / ``upsert_batch`` / ``delete_batch``),
   coalescing consecutive same-kind operations into a single batch call. When
   the staged buffer reduces to *one* such call — all creates, all upserts, or
   all deletes — that call is all-or-nothing on a backend whose batch operations
   are wrapped in a backend transaction (sqlite/postgres/duckdb — those
   reporting :meth:`AsyncDatabase.supports_transactions`).
   :attr:`BufferedTransaction.is_atomic` reports exactly this condition.

What it deliberately does **not** provide:

- **Cross-operation atomicity.** A *mixed* buffer spanning more than one kind
  (e.g. creates *and* deletes, or creates *and* upserts) commits as a
  **sequence** of independent backend batches — one coalesced call per same-kind
  run. If a later batch fails mid-flush, earlier batches have already committed
  and **stay persisted** — a partial commit, with no compensating rollback
  (there cannot be one; the earlier writes are already durable). For such a
  buffer :attr:`~BufferedTransaction.is_atomic` is ``False`` even on a
  transactional backend. There is no public primitive that commits mixed kinds
  all-or-nothing; a consumer needing that guarantee should stage a **single
  kind per transaction** (each single-kind commit is all-or-nothing on a
  transactional backend) and order the per-kind commits so a mid-sequence
  failure leaves a recoverable state.
- **In-transaction isolation or read-your-writes.** Buffered writes are
  invisible to reads (``db.read``) until commit, and concurrent readers never
  see a partially-applied transaction because nothing is written until the
  flush. Do **not** commit two buffered transactions concurrently against a
  single-connection backend (e.g. aiosqlite): the per-batch ``BEGIN`` /
  ``COMMIT`` boundaries the backend issues can interleave. Connection-scoped
  isolation / read-your-writes is not provided — the public API exposes no
  connection-scoped transaction beyond this buffered form. A consumer needing a
  read-modify-write invariant should use optimistic concurrency
  (:meth:`AsyncDatabase.update` / :meth:`AsyncDatabase.upsert` with
  ``expected_version``) or serialize the conflicting work itself (e.g. an
  application-level lock).
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
    :meth:`upsert_batch`, :meth:`delete`) stage operations; nothing reaches the
    backend until :meth:`commit`.
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
        # The *backend* capability is fixed for the life of the handle; whether
        # a given commit is atomic also depends on the staged op composition
        # (see :attr:`is_atomic`), so this is only half the answer.
        self._backend_atomic = bool(db.supports_transactions())

    def _is_single_batch(self) -> bool:
        """Whether the staged ops flush as one coalesced atomic batch.

        True iff every staged op is the *same* batch kind (all ``create``, all
        ``upsert``, or all ``delete``) — the only shape that reduces to a single
        ``create_batch`` / ``upsert_batch`` / ``delete_batch`` call. An empty
        buffer (nothing to fail) and a single op both qualify. Two distinct
        kinds (e.g. create + delete, or create + upsert) flush as separate
        coalesced calls and are not single-batch.
        """
        return len({op[0] for op in self._ops}) <= 1

    @property
    def is_atomic(self) -> bool:
        """Whether :meth:`commit` is all-or-nothing for the **currently staged** ops.

        ``True`` only when the backing backend supports transactions
        (:meth:`AsyncDatabase.supports_transactions`) **and** the staged buffer
        reduces to a single coalesced same-kind batch — all creates, all
        upserts, or all deletes. A buffer spanning more than one kind commits as
        a sequence of independent backend batches and can partially persist on a
        mid-flush failure, so it reports ``False`` even on a transactional
        backend. An empty buffer is trivially atomic.

        Computed from the current ops, so staging more writes can flip a
        previously-atomic handle to non-atomic — read it immediately before
        :meth:`commit` when branching on all-or-nothing semantics.

        After :meth:`commit` or :meth:`rollback` the buffer is empty, so this
        returns ``True`` on a transactional backend — the trivial empty-buffer
        case. That has no operational meaning on a closed handle; check
        :attr:`closed` first if a post-commit read is possible.
        """
        return self._backend_atomic and self._is_single_batch()

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

    async def upsert_batch(self, records: list[Record]) -> None:
        """Stage multiple upserts (the batch sibling of :meth:`create_batch`)."""
        for record in records:
            self._stage(("upsert", (record, None)))

    async def delete(self, id: str) -> None:
        """Stage a delete (flushed via ``delete_batch`` on commit)."""
        self._stage(("delete", id))

    async def commit(self) -> dict[str, int]:
        """Flush all staged writes to the backend.

        Consecutive same-kind operations are coalesced into a single atomic
        batch call (``create_batch`` / ``upsert_batch`` / ``delete_batch``). On
        a backend reporting :meth:`AsyncDatabase.supports_transactions`, each
        coalesced batch is all-or-nothing.

        Atomicity is **per batch**, not across the whole buffer: when the staged
        ops span more than one kind (e.g. mixed create/delete, or create +
        upsert) those batches commit independently, so a mid-flush failure can
        leave earlier batches persisted while a later one is rejected. Check
        :attr:`is_atomic` before relying on all-or-nothing semantics across the
        full commit.

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
                # Coalesce a consecutive upsert run into a single atomic
                # ``upsert_batch`` (mirroring the create/delete coalescing).
                # Both staging forms normalize to a ``Record``: the explicit-id
                # ``upsert(id, record)`` form stamps the id onto the record's
                # ``storage_id`` (which ``upsert_batch`` honors, taking priority
                # over any ``id`` field); the ``upsert(record)`` / staged
                # ``upsert_batch`` form already carries (or mints) its own id.
                records = []
                j = i
                while j < n and ops[j][0] == "upsert":
                    id_or_record, record = ops[j][1]
                    if record is not None:
                        record.storage_id = id_or_record
                        records.append(record)
                    else:
                        records.append(id_or_record)
                    j += 1
                upserted = await self._db.upsert_batch(records)
                affected += len(upserted)
                i = j
        return {"affected_rows": affected}

    async def rollback(self) -> None:
        """Discard all staged writes. Idempotent."""
        self._closed = True
        self._ops.clear()

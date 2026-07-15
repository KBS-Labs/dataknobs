"""Buffered transaction handle for :class:`AsyncDatabase`.

A :class:`BufferedTransaction` defers every write until :meth:`commit`. This
gives two guarantees:

1. **Universal rollback.** Because writes are buffered, raising before commit
   (e.g. from business logic inside an ``async with db.transaction()`` block)
   persists nothing — on *any* backend, transactional or not.
2. **Atomic commit on transactional backends — any composition.** The commit
   flush replays the buffer through the database's atomic batch primitives
   (``create_batch`` / ``upsert_batch`` / ``delete_batch``), coalescing
   consecutive same-kind operations into a single batch call. On a backend whose
   batch operations are wrapped in a backend transaction (sqlite/postgres/duckdb
   — those reporting :meth:`AsyncDatabase.supports_transactions`), the *whole*
   flush runs inside **one** native transaction, so a commit is all-or-nothing
   regardless of composition: a single same-kind batch (all creates, all
   upserts, or all deletes) **and** a mixed buffer spanning several kinds (e.g.
   creates *and* deletes, or creates *and* upserts) alike. A mid-flush failure
   rolls the whole commit back — no partial persistence.
   :attr:`BufferedTransaction.is_atomic` reports this: ``True`` for any
   composition on a transactional backend.

What it deliberately does **not** provide:

- **Atomic commit on a non-transactional backend.** On ``memory`` / ``file`` /
  ``s3`` / ``elasticsearch`` (those reporting
  :meth:`AsyncDatabase.supports_transactions` as ``False``) there is no native
  transaction to span the flush, so a *mixed* buffer commits as a **sequence**
  of independent backend batches — one coalesced call per same-kind run. If a
  later batch fails mid-flush, earlier batches have already been applied and
  **stay applied**, with no compensating rollback. For such a backend
  :attr:`~BufferedTransaction.is_atomic` is ``False`` for any non-empty buffer.
  A consumer needing all-or-nothing there should open the transaction with the
  default ``policy="strict"``, which fails closed with
  :class:`~dataknobs_common.CapabilityNotSupportedError` rather than promising
  atomicity the backend cannot give.
- **In-transaction isolation or read-your-writes.** Buffered writes are
  invisible to reads (``db.read``) until commit, and concurrent readers never
  see a partially-applied transaction because nothing is written until the
  flush. On a single-connection backend (e.g. aiosqlite or duckdb) a multi-kind
  commit holds one open native transaction across several ``await`` boundaries
  (begin, then each coalesced batch, then commit); **no concurrent write of any
  kind** may run against that same instance during the flush — not just another
  buffered commit, but also a plain ``db.create`` / ``db.upsert`` / ``db.delete``
  — because that write would issue its own ``BEGIN`` while the multi-kind
  transaction is already open and the boundaries would interleave. Serialize
  writes to a single-connection instance yourself if they can overlap.
  Connection-scoped isolation / read-your-writes is not provided — the public
  API exposes no connection-scoped transaction beyond this buffered form. A
  consumer needing a
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

    @property
    def is_atomic(self) -> bool:
        """Whether :meth:`commit` is all-or-nothing for the staged ops.

        ``True`` whenever the backing backend supports transactions
        (:meth:`AsyncDatabase.supports_transactions`) — the commit flush spans
        every coalesced batch in one native transaction there, so a buffer of
        *any* composition (single-kind or multi-kind) is all-or-nothing. On a
        non-transactional backend this is ``False`` (there is no native
        transaction to span the flush; a multi-kind buffer can partially
        persist). It therefore reflects the backend capability directly and is
        stable across staging.
        """
        return self._backend_atomic

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

    def _coalesce(self) -> list[tuple[str, list[Any]]]:
        """Reduce the staged ops to a list of coalesced same-kind runs.

        Each element is ``(kind, payload)`` where ``kind`` is
        ``"create"`` / ``"delete"`` / ``"upsert"`` and ``payload`` is the list
        passed to the matching ``*_batch`` call — records for create/upsert, ids
        for delete. Consecutive same-kind ops merge into one run (the shape that
        reduces to a single batch call); a change of kind starts a new run.

        Both upsert staging forms normalize to a ``Record`` here: the explicit-id
        ``upsert(id, record)`` form stamps the id onto the record's
        ``storage_id`` (which ``upsert_batch`` honors, taking priority over any
        ``id`` field); the ``upsert(record)`` / staged ``upsert_batch`` form
        already carries (or mints) its own id.
        """
        runs: list[tuple[str, list[Any]]] = []
        ops = self._ops
        i = 0
        n = len(ops)
        while i < n:
            kind = ops[i][0]
            if kind == "upsert":
                records: list[Any] = []
                j = i
                while j < n and ops[j][0] == "upsert":
                    id_or_record, record = ops[j][1]
                    if record is not None:
                        record.storage_id = id_or_record
                        records.append(record)
                    else:
                        records.append(id_or_record)
                    j += 1
                runs.append(("upsert", records))
                i = j
            else:  # "create" or "delete" — payload is the raw staged value
                payload: list[Any] = []
                j = i
                while j < n and ops[j][0] == kind:
                    payload.append(ops[j][1])
                    j += 1
                runs.append((kind, payload))
                i = j
        return runs

    async def commit(self) -> dict[str, int]:
        """Flush all staged writes to the backend.

        Consecutive same-kind operations are coalesced into a single atomic
        batch call (``create_batch`` / ``upsert_batch`` / ``delete_batch``). On
        a backend reporting :meth:`AsyncDatabase.supports_transactions`, the
        *whole* flush runs inside one native transaction, so the commit is
        all-or-nothing across every batch regardless of composition — a
        multi-kind buffer (e.g. mixed create/delete, or create + upsert) rolls
        back entirely on a mid-flush failure, not partially. On a
        non-transactional backend a multi-kind buffer flushes as independent
        batches and can partially persist; :attr:`is_atomic` reports which case
        applies.

        Idempotent: a second call (after commit or rollback) is a no-op.

        Returns:
            ``{"affected_rows": <count>}``.
        """
        if self._closed:
            return {"affected_rows": 0}
        self._closed = True

        runs = self._coalesce()

        async def replay(tx: Any) -> int:
            affected = 0
            for kind, payload in runs:
                if kind == "create":
                    affected += len(await self._db.create_batch(payload, _tx=tx))
                elif kind == "delete":
                    affected += sum(
                        1 for ok in await self._db.delete_batch(payload, _tx=tx) if ok
                    )
                else:  # "upsert"
                    affected += len(await self._db.upsert_batch(payload, _tx=tx))
            return affected

        # Span the several coalesced batches in one native transaction only for
        # a multi-kind buffer on a transactional backend — the case that would
        # otherwise partially persist. Single-kind (already one native tx),
        # empty, and non-transactional buffers stay on the direct ``_tx=None``
        # path, byte-identical to the per-batch flush.
        if self._backend_atomic and len(runs) > 1:
            async with self._db._transaction() as tx:
                # A backend reporting ``supports_transactions()`` MUST override
                # ``_transaction()`` to yield a real handle (the ABC default
                # yields ``None``). Fail closed on the misimplemented case rather
                # than silently threading ``_tx=None`` into every batch — which
                # would run each on its own boundary, losing the cross-kind
                # atomicity ``is_atomic`` just promised. Nothing has flushed yet,
                # so this raises before any partial persistence.
                if tx is None:
                    raise OperationError(
                        f"{type(self._db).__name__} reports "
                        "supports_transactions() is True but its _transaction() "
                        "yielded no handle; a transactional backend must override "
                        "_transaction() to span a multi-kind commit flush "
                        "atomically."
                    )
                affected = await replay(tx)
        else:
            affected = await replay(None)
        return {"affected_rows": affected}

    async def rollback(self) -> None:
        """Discard all staged writes. Idempotent."""
        self._closed = True
        self._ops.clear()

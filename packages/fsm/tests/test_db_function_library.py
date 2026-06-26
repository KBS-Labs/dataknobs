"""Reproduce-first behavioral tests for the DB function library (170-FU2, PR-A).

Covers the three things PR-A makes real:

* **W0 — record identity.** ``KeyColumnsIdentity`` joins key columns with a
  collision-safe separator, so two rows whose values contain the legacy ``"_"``
  separator no longer derive the same id. ``CallableIdentity`` / the
  ``resolve_identity`` sugar cover the escape hatches.
* **W1 — ``DatabaseBulkInsert.on_duplicate``.** Previously an advertised-but-dead
  knob (the adapter always created). Now ``error`` / ``ignore`` / ``update`` are
  honored against the derived identity, and a dedup policy with no identity is a
  loud ``ConfigurationError`` rather than silent create-only.
* **W2 — ``BatchCommit`` reshape + atomicity policy.** ``BatchCommit`` used to
  call ``resource.transaction()`` — a method no adapter implemented, so it
  crashed on first use. It now routes through the real ``commit_batch`` atomic
  primitive, with a consumer-selected ``atomicity`` policy (``best_effort`` /
  ``require``).

Real constructs only — a real ``AsyncDatabase`` (``file`` / ``sqlite`` /
``memory``) reached through the real ``AsyncDatabaseResourceAdapter``, and real
FSM builds via the ``custom_functions=`` + state ``functions`` idiom. Target
rows are read back through a fresh ``AsyncDatabase.from_backend`` to prove
persistence independently of the writing adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataknobs_common import CapabilityNotSupportedError
from dataknobs_common.exceptions import ConfigurationError, ValidationError
from dataknobs_common.testing import assert_no_blocking
from dataknobs_data import AsyncDatabase
from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.functions.base import ResourceError
from dataknobs_fsm.functions.library.database import (
    BatchCommit,
    DatabaseBulkInsert,
    DatabaseUpsert,
)
from dataknobs_fsm.functions.library.identity import (
    CallableIdentity,
    KeyColumnsIdentity,
    resolve_identity,
)
from dataknobs_fsm.resources.database import AsyncDatabaseResourceAdapter


@pytest.fixture(scope="module", autouse=True)
def _warm_async_backend_registry() -> None:
    """Initialize the lazy async-backend registry outside any detector block.

    The first ``AsyncDatabase.from_backend`` call in a process triggers
    ``_register_async_backends``, which imports every async backend — including
    duckdb, whose import reads its version metadata file synchronously. That
    one-shot, setup-time read is acceptable (it is not on a hot loop), but if it
    first runs *inside* an ``assert_no_blocking()`` block it trips the detector.
    Warming the registry here mirrors what the rest of the suite gets for free
    by running a non-detector path first, so the detector-wrapped tests below
    measure only the code under test.
    """
    import asyncio

    async def _go() -> None:
        db = await AsyncDatabase.from_backend("memory", {"type": "memory"})
        await db.close()

    asyncio.run(_go())


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _file_adapter(tmp_path: Path, name: str) -> AsyncDatabaseResourceAdapter:
    return AsyncDatabaseResourceAdapter(
        name=name, backend="file", path=str(tmp_path / f"{name}.json")
    )


def _sqlite_adapter(tmp_path: Path, name: str) -> AsyncDatabaseResourceAdapter:
    return AsyncDatabaseResourceAdapter(
        name=name, backend="sqlite", path=str(tmp_path / f"{name}.db")
    )


class _CountingAdapter(AsyncDatabaseResourceAdapter):
    """Real adapter that also counts ``commit_batch`` invocations.

    A behaviour-preserving subclass (not a mock) so a test can prove
    ``BatchCommit.batch_size`` actually chunks — outcome assertions alone can't
    distinguish one big commit from several small ones.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.commit_batch_calls = 0

    async def commit_batch(self, records, **kwargs):  # type: ignore[override]
        self.commit_batch_calls += 1
        return await super().commit_batch(records, **kwargs)


def _ctx(resource_name: str, adapter: AsyncDatabaseResourceAdapter) -> dict:
    """A plain-dict FunctionContext stand-in carrying the injected resource."""
    return {"resources": {resource_name: adapter}}


async def _read_file(path: Path, record_id: str) -> Any:
    db = await AsyncDatabase.from_backend("file", {"type": "file", "path": str(path)})
    try:
        return await db.read(record_id)
    finally:
        await db.close()


async def _count_file(path: Path) -> int:
    db = await AsyncDatabase.from_backend("file", {"type": "file", "path": str(path)})
    try:
        return await db.count()
    finally:
        await db.close()


# --------------------------------------------------------------------------- #
# W0 — record identity
# --------------------------------------------------------------------------- #
def test_key_columns_identity_is_collision_safe() -> None:
    """Composite-key rows whose values contain the legacy '_' separator must
    derive *distinct* ids under the default unit separator — the latent
    corruption the old ``"_".join`` rule allowed.
    """
    ident = KeyColumnsIdentity(["k1", "k2"])
    row_a = {"k1": "a_b", "k2": "c"}
    row_b = {"k1": "a", "k2": "b_c"}
    assert ident.derive(row_a) != ident.derive(row_b)
    # The legacy underscore separator collides them — proving the regression
    # the default separator fixes.
    legacy = KeyColumnsIdentity(["k1", "k2"], sep="_")
    assert legacy.derive(row_a) == legacy.derive(row_b)


def test_callable_identity_overrides_derivation() -> None:
    ident = CallableIdentity(lambda row: f"user:{row['email']}")
    assert ident.derive({"email": "a@b.c"}) == "user:a@b.c"


def test_resolve_identity_picks_one_strategy() -> None:
    assert resolve_identity() is None
    assert isinstance(resolve_identity(key_columns=["id"]), KeyColumnsIdentity)
    assert isinstance(
        resolve_identity(id_fn=lambda r: r["id"]), CallableIdentity
    )
    with pytest.raises(ConfigurationError):
        resolve_identity(key_columns=["id"], id_fn=lambda r: r["id"])


def test_empty_key_columns_yields_no_identity() -> None:
    assert KeyColumnsIdentity([]).derive({"a": 1}) is None


def test_key_columns_identity_missing_or_none_column_raises() -> None:
    """A key column absent from the row, or present-but-``None``, has no
    well-defined value — deriving an id from it would let every such row
    collide (with each other and with a genuine ``"None"`` value). Fail closed
    instead of silently rendering ``"None"``.
    """
    ident = KeyColumnsIdentity(["k1", "k2"])
    with pytest.raises(ValidationError):
        ident.derive({"k1": "a"})  # k2 absent
    with pytest.raises(ValidationError):
        ident.derive({"k1": "a", "k2": None})  # k2 present but None
    # A row with all components present still derives fine.
    assert ident.derive({"k1": "a", "k2": "b"}) == "a\x1fb"


# --------------------------------------------------------------------------- #
# W1 — DatabaseBulkInsert.on_duplicate
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_bulk_insert_on_duplicate_error(tmp_path: Path) -> None:
    ident = KeyColumnsIdentity(["id"])
    adapter = _file_adapter(tmp_path, "err")
    try:
        await adapter.bulk_insert(
            "t", [{"id": "1", "v": "a"}], identity=ident, on_duplicate="error"
        )
        with pytest.raises(ResourceError):
            await adapter.bulk_insert(
                "t", [{"id": "1", "v": "b"}], identity=ident, on_duplicate="error"
            )
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_bulk_insert_on_duplicate_ignore_skips(tmp_path: Path) -> None:
    ident = KeyColumnsIdentity(["id"])
    adapter = _file_adapter(tmp_path, "ign")
    try:
        await adapter.bulk_insert(
            "t", [{"id": "1", "v": "a"}], identity=ident, on_duplicate="ignore"
        )
        res = await adapter.bulk_insert(
            "t", [{"id": "1", "v": "b"}], identity=ident, on_duplicate="ignore"
        )
        assert res["affected_rows"] == 0
    finally:
        await adapter.aclose()
    record = await _read_file(tmp_path / "ign.json", "1")
    assert record is not None and record.to_dict()["v"] == "a"


@pytest.mark.asyncio
async def test_bulk_insert_on_duplicate_update_overwrites(tmp_path: Path) -> None:
    ident = KeyColumnsIdentity(["id"])
    adapter = _file_adapter(tmp_path, "upd")
    try:
        await adapter.bulk_insert(
            "t", [{"id": "1", "v": "a"}], identity=ident, on_duplicate="update"
        )
        res = await adapter.bulk_insert(
            "t", [{"id": "1", "v": "b"}], identity=ident, on_duplicate="update"
        )
        assert res["affected_rows"] == 1
    finally:
        await adapter.aclose()
    record = await _read_file(tmp_path / "upd.json", "1")
    assert record is not None and record.to_dict()["v"] == "b"


@pytest.mark.asyncio
async def test_bulk_insert_no_identity_creates_all(tmp_path: Path) -> None:
    adapter = _file_adapter(tmp_path, "create")
    try:
        res = await adapter.bulk_insert(
            "t", [{"v": "a"}, {"v": "b"}]
        )
        assert res["affected_rows"] == 2
    finally:
        await adapter.aclose()
    assert await _count_file(tmp_path / "create.json") == 2


def test_bulk_insert_dedup_without_identity_is_configuration_error() -> None:
    """A dedup policy with nothing to dedup against is a loud misconfig, not a
    silent fallback to create-only.
    """
    with pytest.raises(ConfigurationError):
        DatabaseBulkInsert("r", "t", on_duplicate="ignore")
    with pytest.raises(ConfigurationError):
        DatabaseBulkInsert("r", "t", on_duplicate="update")
    with pytest.raises(ConfigurationError):
        DatabaseBulkInsert("r", "t", on_duplicate="bogus")


def test_bulk_insert_nonpositive_chunk_size_is_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        DatabaseBulkInsert("r", "t", chunk_size=0)
    with pytest.raises(ConfigurationError):
        DatabaseBulkInsert("r", "t", chunk_size=-5)


# --------------------------------------------------------------------------- #
# W1b — DatabaseUpsert.on_conflict (sibling of the on_duplicate dead-knob)
# --------------------------------------------------------------------------- #
def test_upsert_conflict_without_identity_is_configuration_error() -> None:
    """``error`` / ``ignore`` need an id to detect the conflict against; with no
    identity they would silently degrade to create-only — the same dead knob
    DatabaseBulkInsert fails closed on.
    """
    with pytest.raises(ConfigurationError):
        DatabaseUpsert("r", "t", on_conflict="error")
    with pytest.raises(ConfigurationError):
        DatabaseUpsert("r", "t", on_conflict="ignore")
    with pytest.raises(ConfigurationError):
        DatabaseUpsert("r", "t", on_conflict="bogus")


def test_upsert_update_without_identity_is_allowed() -> None:
    """The default ``update`` with no identity is a legitimate plain create —
    it must NOT be guarded (would break the no-key upsert use case).
    """
    fn = DatabaseUpsert("r", "t")  # on_conflict="update", no identity
    assert fn.identity is None
    # An identity-bearing conflict policy is likewise fine.
    DatabaseUpsert("r", "t", key_columns=["id"], on_conflict="error")


# --------------------------------------------------------------------------- #
# W1c — identity ValidationError must surface through .transform() with its own
#       type, not get masked as a generic TransformError (sibling of the
#       BatchCommit passthrough fix)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_upsert_transform_surfaces_validation_error(tmp_path: Path) -> None:
    """A row missing a key column raises ``ValidationError`` during identity
    derivation inside ``resource.upsert``. ``DatabaseUpsert.transform`` must let
    it propagate as ``ValidationError`` — masking it as ``TransformError`` would
    hide a consumer-actionable signal (the exact gap the BatchCommit passthrough
    closes for batch commits).
    """
    adapter = _file_adapter(tmp_path, "upv")
    fn = DatabaseUpsert("target_db", "t", key_columns=["id"])
    try:
        with pytest.raises(ValidationError):
            await fn.transform(
                {"record": {"v": "a"}},  # no "id" → identity derivation fails
                _ctx("target_db", adapter),
            )
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_bulk_insert_transform_surfaces_validation_error(
    tmp_path: Path,
) -> None:
    """Sibling of the upsert case: a row missing a key column must surface
    ``ValidationError`` through ``DatabaseBulkInsert.transform`` rather than the
    generic ``TransformError`` wrapper.
    """
    adapter = _file_adapter(tmp_path, "biv")
    fn = DatabaseBulkInsert(
        "target_db", "t", key_columns=["id"], on_duplicate="update"
    )
    try:
        with pytest.raises(ValidationError):
            await fn.transform(
                {"records": [{"v": "a"}]},  # no "id"
                _ctx("target_db", adapter),
            )
    finally:
        await adapter.aclose()


# --------------------------------------------------------------------------- #
# W2 — BatchCommit reshape + atomicity policy
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_commit_batch_create_mode_persists(tmp_path: Path) -> None:
    adapter = _file_adapter(tmp_path, "cb")
    try:
        res = await adapter.commit_batch([{"v": "a"}, {"v": "b"}, {"v": "c"}])
        assert res["affected_rows"] == 3
    finally:
        await adapter.aclose()
    assert await _count_file(tmp_path / "cb.json") == 3


@pytest.mark.asyncio
async def test_commit_batch_identity_is_idempotent(tmp_path: Path) -> None:
    ident = KeyColumnsIdentity(["id"])
    adapter = _file_adapter(tmp_path, "idem")
    batch = [{"id": "1", "v": "a"}, {"id": "2", "v": "b"}]
    try:
        await adapter.commit_batch(batch, identity=ident)
        await adapter.commit_batch(batch, identity=ident)  # re-commit
    finally:
        await adapter.aclose()
    # Idempotent: re-commit upserts the same two ids rather than duplicating.
    assert await _count_file(tmp_path / "idem.json") == 2


@pytest.mark.asyncio
async def test_commit_batch_empty_is_noop(tmp_path: Path) -> None:
    adapter = _file_adapter(tmp_path, "empty")
    try:
        res = await adapter.commit_batch([])
        assert res["affected_rows"] == 0
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_commit_batch_require_raises_on_non_transactional(tmp_path: Path) -> None:
    adapter = _file_adapter(tmp_path, "req")
    try:
        with pytest.raises(CapabilityNotSupportedError):
            await adapter.commit_batch([{"v": "a"}], atomicity="require")
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_commit_batch_require_succeeds_on_sqlite(tmp_path: Path) -> None:
    adapter = _sqlite_adapter(tmp_path, "sq")
    try:
        res = await adapter.commit_batch(
            [{"v": "a"}, {"v": "b"}], atomicity="require"
        )
        assert res["affected_rows"] == 2
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_commit_batch_unknown_atomicity_is_configuration_error(
    tmp_path: Path,
) -> None:
    adapter = _file_adapter(tmp_path, "bad")
    try:
        with pytest.raises(ConfigurationError):
            await adapter.commit_batch([{"v": "a"}], atomicity="bogus")
    finally:
        await adapter.aclose()


# --------------------------------------------------------------------------- #
# W2b — BatchCommit.transform: batch is cleared, committed_count wins, and
#       batch_size actually chunks
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_batch_commit_transform_clears_batch_and_overrides_count(
    tmp_path: Path,
) -> None:
    """The documented post-condition: after a commit ``batch`` is emptied and
    ``committed_count`` reflects this commit. Reproduces the override-ordering
    bug where ``**data`` (spread last) clobbered both back to their inputs.
    """
    adapter = _file_adapter(tmp_path, "clear")
    fn = BatchCommit("target_db")
    try:
        result = await fn.transform(
            {
                "batch": [{"v": "a"}, {"v": "b"}],
                "committed_count": 999,  # stale value from a prior step
                "other": "keep-me",
            },
            _ctx("target_db", adapter),
        )
    finally:
        await adapter.aclose()
    assert result["batch"] == []  # cleared (was the original 2-row batch)
    assert result["committed_count"] == 2  # new outcome, not the stale 999
    assert result["other"] == "keep-me"  # unrelated passthrough preserved
    assert await _count_file(tmp_path / "clear.json") == 2


@pytest.mark.asyncio
async def test_batch_commit_batch_size_chunks_best_effort(tmp_path: Path) -> None:
    """``batch_size`` bounds each best_effort commit: 5 rows at batch_size=2
    must reach the adapter as 3 ``commit_batch`` calls (not silently inert).
    """
    adapter = _CountingAdapter(
        name="chunk", backend="file", path=str(tmp_path / "chunk.json")
    )
    fn = BatchCommit("target_db", batch_size=2)
    batch = [{"v": str(i)} for i in range(5)]
    try:
        result = await fn.transform({"batch": batch}, _ctx("target_db", adapter))
    finally:
        await adapter.aclose()
    assert adapter.commit_batch_calls == 3  # ceil(5 / 2)
    assert result["committed_count"] == 5
    assert result["batch"] == []
    assert await _count_file(tmp_path / "chunk.json") == 5


@pytest.mark.asyncio
async def test_batch_commit_require_issues_single_atomic_batch(
    tmp_path: Path,
) -> None:
    """Under ``atomicity="require"`` the commit must be one all-or-nothing
    batch — ``batch_size`` does not chunk it, so a 5-row batch at batch_size=2
    is a SINGLE ``commit_batch`` call (whole-batch atomicity wins).
    """
    adapter = _CountingAdapter(
        name="atomic", backend="sqlite", path=str(tmp_path / "atomic.db")
    )
    fn = BatchCommit("target_db", batch_size=2, atomicity="require")
    batch = [{"v": str(i)} for i in range(5)]
    try:
        result = await fn.transform({"batch": batch}, _ctx("target_db", adapter))
    finally:
        await adapter.aclose()
    assert adapter.commit_batch_calls == 1  # NOT chunked under "require"
    assert result["committed_count"] == 5
    assert result["batch"] == []


def test_batch_commit_nonpositive_batch_size_is_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        BatchCommit("r", batch_size=0)
    with pytest.raises(ConfigurationError):
        BatchCommit("r", batch_size=-1)


# --------------------------------------------------------------------------- #
# Full-FSM behavioral coverage (mirrors the W2 reference idiom)
# --------------------------------------------------------------------------- #
def _single_state_fsm(
    name: str, func, target_cfg: dict[str, Any]
) -> AsyncSimpleFSM:
    config = {
        "name": name,
        "data_mode": DataHandlingMode.COPY.value,
        "resources": [
            {"name": "target_db", "type": "async_database", "config": target_cfg},
        ],
        "states": [
            {
                "name": "load",
                "is_start": True,
                "resources": ["target_db"],
                "functions": {"transform": {"type": "registered", "name": "fn"}},
            },
            {"name": "done", "is_end": True},
        ],
        "arcs": [{"from": "load", "to": "done", "name": "loaded"}],
    }
    return AsyncSimpleFSM(
        config,
        data_mode=DataHandlingMode.COPY,
        custom_functions={"fn": func},
    )


@pytest.mark.asyncio
async def test_batch_commit_through_fsm_persists(tmp_path: Path) -> None:
    target = {"type": "file", "path": str(tmp_path / "fsm_batch.json")}
    fsm = _single_state_fsm("batch", BatchCommit("target_db"), target)
    try:
        with assert_no_blocking():
            result = await fsm.process(
                {"batch": [{"v": "a"}, {"v": "b"}, {"v": "c"}]}
            )
        assert result["success"], f"FSM did not complete cleanly: {result}"
    finally:
        await fsm.close()
    assert await _count_file(tmp_path / "fsm_batch.json") == 3


@pytest.mark.asyncio
async def test_bulk_insert_through_fsm_persists(tmp_path: Path) -> None:
    target = {"type": "file", "path": str(tmp_path / "fsm_bulk.json")}
    fsm = _single_state_fsm(
        "bulk", DatabaseBulkInsert("target_db", "rows"), target
    )
    try:
        with assert_no_blocking():
            result = await fsm.process({"records": [{"v": "a"}, {"v": "b"}]})
        assert result["success"], f"FSM did not complete cleanly: {result}"
    finally:
        await fsm.close()
    assert await _count_file(tmp_path / "fsm_bulk.json") == 2

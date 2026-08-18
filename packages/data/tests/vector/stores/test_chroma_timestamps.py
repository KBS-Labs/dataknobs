"""``ChromaVectorStore`` timestamp tracking, and the in-band storage of it.

Cross-backend timestamp semantics are pinned by the parity suite, which
this backend now joins. What is left is what only this backend can get
wrong: it has no side-car to keep timestamps in, so it keeps them in the
collection metadata — the one namespace the consumer also owns.

Two properties follow, and neither is visible from the parity suite.
The stored keys must never reach a consumer, through any read path; and
the stored form must stay readable when the configured output format
changes, because format is an output concern and rows outlive it.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

import numpy as np
import pytest
from dataknobs_common.testing import is_chromadb_available

from dataknobs_data.testing import chroma_embedding_function

if is_chromadb_available():
    from dataknobs_data.vector.stores.chroma import ChromaVectorStore

requires_chromadb = pytest.mark.skipif(not is_chromadb_available(), reason="chromadb not installed")

pytestmark = [pytest.mark.asyncio, requires_chromadb]

DIMENSIONS = 4
RESERVED_PREFIX = "\x00dk\x00"


def _vec(i: int = 0) -> np.ndarray:
    v = np.zeros(DIMENSIONS, dtype=np.float32)
    v[i % DIMENSIONS] = 1.0
    return v


async def _store(**config: Any) -> Any:
    store = ChromaVectorStore(
        {
            "dimensions": DIMENSIONS,
            "collection_name": f"test_chroma_ts_{uuid.uuid4().hex[:8]}",
            **config,
        }
    )
    await store.initialize()
    return store


async def _drop(store: Any) -> None:
    try:
        store.client.delete_collection(name=store.collection_name)
    finally:
        await store.close()


def _reserved_in(meta: dict[str, Any] | None) -> list[str]:
    return [k for k in (meta or {}) if k.startswith(RESERVED_PREFIX)]


async def test_no_read_path_exposes_a_reserved_key():  # type: ignore[no-untyped-def]
    """The storage keys stay out of every public read.

    Enumerated rather than sampled: the strip lives in one place, and
    the reason it lives there is that seven call sites decode. A new
    read path that decodes inherits the strip; one that reaches raw
    chromadb metadata would not, and this is what would catch it.
    """
    store = await _store()
    try:
        await store.add_vectors(
            [_vec(0), _vec(1)], ids=["a", "b"], metadata=[{"g": "x"}, {"g": "y"}]
        )

        _, meta = (await store.get_vectors(["a"]))[0]
        assert _reserved_in(meta) == [], meta

        hits = await store.search(_vec(0), k=2)
        for _, _, m in hits:
            assert _reserved_in(m) == [], m

        with_ts = (await store.get_vectors(["a"], include_timestamps=True))[0][1]
        assert _reserved_in(with_ts) == [], with_ts

        fields = await store.metadata_fields()
        assert [f for f in fields if f.startswith(RESERVED_PREFIX)] == [], fields
        assert fields == {"g"}

        # The residual post-filter sees decoded metadata, so a reserved
        # key can neither match nor be matched against.
        assert await store.count(filter={"g": "x"}) == 1
        await store.clear(filter={"g": "y"})
        assert await store.count() == 1

        # The "matched against" half, which the sentence above claimed
        # and nothing checked. A filter naming a reserved key finds
        # nothing, whatever value it asks for — the key is not in the
        # decoded metadata the filter runs against, and a missing key
        # fails a filter. It must not accidentally match on the stored
        # epoch float either.
        stored_created = (
            await asyncio.to_thread(store.collection.get, ids=["a"], include=["metadatas"])
        )["metadatas"][0][ChromaVectorStore._TS_CREATED_KEY]
        assert await store.count(filter={ChromaVectorStore._TS_CREATED_KEY: stored_created}) == 0
        assert await store.count(filter={ChromaVectorStore._TS_UPDATED_KEY: "anything"}) == 0
    finally:
        await _drop(store)


async def test_documents_read_path_is_clean_too():  # type: ignore[no-untyped-def]
    """``search_documents`` decodes through the same boundary.

    The only case in this file that embeds text, so the only one that
    needs an embedding function. The deterministic one keeps chromadb
    from reaching for its default, which downloads a model on first use
    and would fail — not skip — on a runner with a cold cache.
    """
    store = await _store(embedding_function=chroma_embedding_function(DIMENSIONS))
    try:
        await store.add_documents(["some text"], ids=["d1"], metadata=[{"g": "x"}])
        hits = await store.search_documents("some text", k=1)
        assert hits[0][0] == "d1"
        assert _reserved_in(hits[0][3]) == [], hits[0][3]
    finally:
        await _drop(store)


async def test_update_metadata_preserves_created_and_advances_updated():  # type: ignore[no-untyped-def]
    """A metadata replacement must not erase the row's timestamps.

    The replacement payload tombstones every stored key the caller
    omitted, and the reserved keys are stored keys no caller supplies —
    so without re-stamping, replacing metadata would silently untrack
    the row.
    """
    store = await _store(timestamps={"format": "datetime"})
    try:
        await store.add_vectors([_vec(0)], ids=["a"], metadata=[{"keep": "K", "drop": "D"}])
        before = (await store.get_vectors(["a"], include_timestamps=True))[0][1]
        assert before is not None

        await asyncio.sleep(0.01)
        assert await store.update_metadata(["a"], [{"keep": "K2"}]) == 1

        after = (await store.get_vectors(["a"], include_timestamps=True))[0][1]
        assert after is not None
        assert after["keep"] == "K2"
        assert "drop" not in after
        assert after["_created_at"] == before["_created_at"]
        assert after["_updated_at"] > before["_updated_at"]
    finally:
        await _drop(store)


async def test_update_metadata_where_preserves_timestamps():  # type: ignore[no-untyped-def]
    """Timestamps survive the decode -> merge -> encode round trip.

    The most likely place to lose them: the merge works from decoded
    metadata, which by construction no longer carries the reserved
    keys, so a re-encode that did not put them back would drop them.
    """
    store = await _store(timestamps={"format": "datetime"})
    try:
        await store.add_vectors([_vec(0)], ids=["a"], metadata=[{"g": "x"}])
        before = (await store.get_vectors(["a"], include_timestamps=True))[0][1]
        assert before is not None

        await asyncio.sleep(0.01)
        assert await store.update_metadata_where({"g": "x"}, {"_stale": True}) == 1

        after = (await store.get_vectors(["a"], include_timestamps=True))[0][1]
        assert after is not None
        assert after["_stale"] is True
        assert after["_created_at"] == before["_created_at"]
        assert after["_updated_at"] > before["_updated_at"]
    finally:
        await _drop(store)


async def test_a_row_written_without_tracking_reports_none():  # type: ignore[no-untyped-def]
    """No backfill for rows that predate tracking.

    Written straight through chromadb so the row genuinely has no
    reserved keys, which is what a collection from an earlier version
    looks like.
    """
    store = await _store()
    try:
        await asyncio.to_thread(
            store.collection.add,
            embeddings=[_vec(0).tolist()],
            ids=["legacy"],
            metadatas=[{"g": "x"}],
        )
        meta = (await store.get_vectors(["legacy"], include_timestamps=True))[0][1]
        assert meta is not None
        assert meta["g"] == "x"
        assert meta["_created_at"] is None
        assert meta["_updated_at"] is None

        # And the next write populates them.
        await store.add_vectors([_vec(0)], ids=["legacy"], metadata=[{"g": "x"}])
        refreshed = (await store.get_vectors(["legacy"], include_timestamps=True))[0][1]
        assert refreshed is not None
        assert refreshed["_created_at"] is not None
    finally:
        await _drop(store)


async def test_a_batch_mixing_new_and_existing_ids_stamps_each_correctly():  # type: ignore[no-untyped-def]
    """One write, two different answers, decided per row.

    ``add_vectors`` reads the batch's stored ``created_at`` values in a
    single ``collection.get``, then stamps row by row. A partially
    overlapping batch is where that per-row lookup earns its keep: the
    existing id must keep its original creation date while the new one
    starts now, and a stamp derived from the batch rather than the row
    would give both the same answer.
    """
    store = await _store()
    try:
        await store.add_vectors([_vec(0)], ids=["old"], metadata=[{"g": "x"}])
        first = (await store.get_vectors(["old"], include_timestamps=True))[0][1]
        assert first is not None
        original_created = first["_created_at"]

        await store.add_vectors(
            [_vec(1), _vec(2)], ids=["old", "fresh"], metadata=[{"g": "y"}, {"g": "z"}]
        )

        rows = await store.get_vectors(["old", "fresh"], include_timestamps=True)
        old_meta, fresh_meta = rows[0][1], rows[1][1]
        assert old_meta is not None and fresh_meta is not None
        assert old_meta["_created_at"] == original_created, "upsert lost the original date"
        assert old_meta["g"] == "y", "the upsert itself must still have applied"
        assert fresh_meta["_created_at"] is not None
    finally:
        await _drop(store)


async def test_consumer_metadata_carrying_a_reserved_key_cannot_corrupt_tracking():  # type: ignore[no-untyped-def]
    """A reserved key supplied by a consumer is overwritten, not honoured.

    Practically unreachable — the keys are NUL-delimited — but the
    failure mode if it were reachable is that a consumer could forge a
    row's creation date, or write a non-numeric value where the decoder
    expects a float. The stored value must be the store's own, and the
    key must not reach any read.
    """
    store = await _store()
    try:
        await store.add_vectors(
            [_vec(0)],
            ids=["a"],
            metadata=[{"g": "x", ChromaVectorStore._TS_CREATED_KEY: "forged"}],
        )

        meta = (await store.get_vectors(["a"], include_timestamps=True))[0][1]
        assert meta is not None
        assert _reserved_in(meta) == [], meta
        assert meta["_created_at"] is not None, "a forged value displaced the real one"

        raw = (await asyncio.to_thread(store.collection.get, ids=["a"], include=["metadatas"]))[
            "metadatas"
        ][0]
        assert isinstance(raw[ChromaVectorStore._TS_CREATED_KEY], int | float)
    finally:
        await _drop(store)


async def _write_legacy_row(store: Any, row_id: str = "legacy") -> None:
    """Add a row straight through chromadb, so it carries no reserved keys.

    What a collection written by a version before this backend tracked
    timestamps looks like.
    """
    await asyncio.to_thread(
        store.collection.add,
        embeddings=[_vec(0).tolist()],
        ids=[row_id],
        metadatas=[{"g": "x"}],
    )


@pytest.mark.parametrize("via", ["update_metadata", "update_metadata_where"])
async def test_a_reserved_key_cannot_be_forged_onto_an_untracked_row(via: str):  # type: ignore[no-untyped-def]
    """The stamp is not what keeps a consumer value out of storage.

    Tracking is established by a write, so ``_stamp`` returns early on a
    row that has none — and that early return was the only thing
    removing a consumer-supplied reserved key from the payload. On a
    pre-tracking row it never ran, so the value went to storage as the
    row's real ``created_at`` and every later read believed it.

    The neighbouring case passes for a reason that does not generalise:
    a *string* forgery is inert because the decoder rejects a
    non-numeric value, and a tracked row's stamp overwrites the key
    anyway. A numeric value on an untracked row meets neither defence.
    So the key is dropped where every write path already funnels —
    encoding to chromadb's contract — rather than left to a downstream
    step that is allowed to do nothing.
    """
    store = await _store()
    try:
        await _write_legacy_row(store)
        forged = {"g": "y", ChromaVectorStore._TS_CREATED_KEY: 0.0}
        if via == "update_metadata":
            await store.update_metadata(["legacy"], [forged])
        else:
            await store.update_metadata_where({"g": "x"}, forged)

        meta = (await store.get_vectors(["legacy"], include_timestamps=True))[0][1]
        assert meta is not None
        assert meta["_created_at"] is None, (
            f"a consumer value became the row's creation date: {meta['_created_at']}"
        )
        assert _reserved_in(meta) == [], meta

        raw = (
            await asyncio.to_thread(store.collection.get, ids=["legacy"], include=["metadatas"])
        )["metadatas"][0]
        assert ChromaVectorStore._TS_CREATED_KEY not in raw, (
            f"the forged key reached storage: {raw}"
        )
    finally:
        await _drop(store)


@pytest.mark.parametrize("via", ["update_metadata", "update_metadata_where"])
async def test_updating_a_legacy_row_does_not_invent_a_created_at(via: str):  # type: ignore[no-untyped-def]
    """An update must not begin tracking a row that was never tracked.

    ``created_at`` for a pre-tracking row is ``None`` — meaning *not
    known* — and the other three backends leave it that way through an
    update: Memory and FAISS guard on the row having a side-car entry,
    and pgvector's ``created_at`` column stays NULL. Only a full write
    establishes tracking.

    Stamping it here would put the *update* time in ``created_at``,
    which is not merely wrong but unrecoverably so: a consumer
    migrating an old collection with one
    ``update_metadata_where(None, ...)`` sweep would set every legacy
    row's creation date to the moment of the sweep, and nothing
    afterwards could distinguish a fabricated date from a real one.
    """
    store = await _store()
    try:
        await _write_legacy_row(store)

        if via == "update_metadata":
            assert await store.update_metadata(["legacy"], [{"g": "y"}]) == 1
        else:
            assert await store.update_metadata_where(None, {"g": "y"}) == 1

        meta = (await store.get_vectors(["legacy"], include_timestamps=True))[0][1]
        assert meta is not None
        assert meta["g"] == "y", "the update itself must still have applied"
        assert meta["_created_at"] is None, f"invented a creation date: {meta}"
    finally:
        await _drop(store)


async def test_replacing_a_bare_legacy_row_with_nothing_is_not_an_error():  # type: ignore[no-untyped-def]
    """The one input that can produce a payload chromadb refuses.

    Every key set here is empty at once: a legacy row (no reserved
    timestamp keys) that also carries no consumer metadata, replaced by
    an empty dict. There is nothing to write and nothing to tombstone,
    and chromadb rejects an empty update dict outright — so the write
    has to be skipped rather than sent.

    Only reachable now that an update declines to start tracking an
    untracked row: while the timestamps were stamped unconditionally the
    payload always had two keys in it and could never be empty.
    """
    store = await _store()
    try:
        await asyncio.to_thread(
            store.collection.add,
            embeddings=[_vec(0).tolist()],
            ids=["bare_legacy"],
        )

        assert await store.update_metadata(["bare_legacy"], [{}]) == 1

        meta = (await store.get_vectors(["bare_legacy"], include_timestamps=True))[0][1]
        assert meta is not None
        assert meta["_created_at"] is None
    finally:
        await _drop(store)


async def test_updating_a_tracked_row_still_preserves_created_at():  # type: ignore[no-untyped-def]
    """The guard above must not cost a tracked row its real ``created_at``.

    Same code path, opposite input: a row that *does* carry a stored
    creation date keeps it across the same update.
    """
    store = await _store()
    try:
        await store.add_vectors([_vec(0)], ids=["r1"], metadata=[{"g": "x"}])
        before = (await store.get_vectors(["r1"], include_timestamps=True))[0][1]
        assert before is not None and before["_created_at"] is not None

        await store.update_metadata(["r1"], [{"g": "y"}])

        after = (await store.get_vectors(["r1"], include_timestamps=True))[0][1]
        assert after is not None
        assert after["_created_at"] == before["_created_at"]
        assert after["_updated_at"] is not None
    finally:
        await _drop(store)


async def test_consumer_metadata_wins_a_key_collision():  # type: ignore[no-untyped-def]
    """The documented collision policy holds here as elsewhere.

    Keeping the storage key distinct from the output key is what makes
    this expressible: the consumer's ``_created_at`` is ordinary
    metadata and the store's is not, so the two are still telling apart
    at injection time.
    """
    store = await _store()
    try:
        await store.add_vectors([_vec(0)], ids=["a"], metadata=[{"_created_at": "mine"}])
        meta = (await store.get_vectors(["a"], include_timestamps=True))[0][1]
        assert meta is not None
        assert meta["_created_at"] == "mine"
        # The store's own tracking is unaffected and still reported.
        assert meta["_updated_at"] is not None
    finally:
        await _drop(store)


async def test_no_injection_without_metadata():  # type: ignore[no-untyped-def]
    """``include_metadata=False`` suppresses injection rather than forcing a dict."""
    store = await _store()
    try:
        await store.add_vectors([_vec(0)], ids=["a"], metadata=[{"g": "x"}])
        rows = await store.get_vectors(["a"], include_metadata=False, include_timestamps=True)
        assert rows[0][1] is None
        hits = await store.search(_vec(0), k=1, include_metadata=False, include_timestamps=True)
        assert hits[0][2] is None
    finally:
        await _drop(store)


@pytest.mark.parametrize("fmt,expected", [("iso", str), ("epoch", float), ("datetime", datetime)])
async def test_every_format_renders_from_the_stored_value(fmt: str, expected: type):  # type: ignore[no-untyped-def]
    """Format is an output concern applied to one stored representation."""
    store = await _store(timestamps={"format": fmt})
    try:
        await store.add_vectors([_vec(0)], ids=["a"], metadata=[{"g": "x"}])
        meta = (await store.get_vectors(["a"], include_timestamps=True))[0][1]
        assert meta is not None
        assert isinstance(meta["_created_at"], expected)
        assert isinstance(meta["_updated_at"], expected)
    finally:
        await _drop(store)


async def test_a_row_outlives_a_change_of_configured_format():  # type: ignore[no-untyped-def]
    """A store reconfigured to another format still reads its old rows.

    This is why the stored form does not follow ``timestamps.format``.
    Had the row been written in the configured format, changing that
    config would have made every existing row unreadable — the failure
    the fixed stored form exists to prevent.
    """
    collection = f"test_chroma_ts_{uuid.uuid4().hex[:8]}"
    writer = ChromaVectorStore(
        {"dimensions": DIMENSIONS, "collection_name": collection, "timestamps": {"format": "iso"}}
    )
    await writer.initialize()
    await writer.add_vectors([_vec(0)], ids=["a"], metadata=[{"g": "x"}])
    written = (await writer.get_vectors(["a"], include_timestamps=True))[0][1]
    assert written is not None and isinstance(written["_created_at"], str)
    await writer.close()

    reader = ChromaVectorStore(
        {"dimensions": DIMENSIONS, "collection_name": collection, "timestamps": {"format": "epoch"}}
    )
    await reader.initialize()
    try:
        meta = (await reader.get_vectors(["a"], include_timestamps=True))[0][1]
        assert meta is not None
        assert isinstance(meta["_created_at"], float)
        assert datetime.fromisoformat(written["_created_at"]).timestamp() == pytest.approx(
            meta["_created_at"]
        )
    finally:
        await _drop(reader)


async def test_reserved_keys_coexist_with_the_nonscalar_encoding():  # type: ignore[no-untyped-def]
    """Both in-band conventions occupy the same dict without interfering."""
    store = await _store()
    try:
        payload = {"tags": ["a", "b"], "nested": {"k": 1}, "empty": []}
        await store.add_vectors([_vec(0)], ids=["a"], metadata=[payload])
        meta = (await store.get_vectors(["a"], include_timestamps=True))[0][1]
        assert meta is not None
        assert meta["tags"] == ["a", "b"]
        assert meta["nested"] == {"k": 1}
        assert meta["empty"] == []
        assert meta["_created_at"] is not None
        assert _reserved_in(meta) == []
    finally:
        await _drop(store)

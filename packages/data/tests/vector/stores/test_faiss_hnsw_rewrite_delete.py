"""``FaissVectorStore`` over an HNSW index rewrites and deletes the ids it holds.

FAISS does not implement ``remove_ids`` for HNSW, and every path that
evicts an id called it: ``add_vectors`` evicting a re-added id (the upsert
contract), ``delete_vectors``, and ``clear(filter=...)`` through it. So a
second write of any id raised ``RuntimeError``, and a delete did worse than
refuse: it dropped the id from the store's own maps *before* calling
``remove_ids``, so after the raise ``get_vectors`` read the id as absent
while ``count()`` still counted it and ``search()`` still ranked it, under
its internal id rather than the caller's.

Flat and IVF indexes answer every one of these already; the tests run the
same assertions over ``flat`` as a control.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from dataknobs_common.testing import is_faiss_available

if is_faiss_available():
    from dataknobs_data.vector.stores.faiss import FaissVectorStore

requires_faiss = pytest.mark.skipif(not is_faiss_available(), reason="faiss not installed")

pytestmark = [pytest.mark.asyncio, requires_faiss]

DIMENSIONS = 8
INDEX_TYPES = ["hnsw", "flat"]


def _row(axis: int) -> np.ndarray:
    row = np.zeros(DIMENSIONS, dtype=np.float32)
    row[axis] = 1.0
    return row


async def _store(index_type: str, **config: object) -> FaissVectorStore:
    store = FaissVectorStore({"dimensions": DIMENSIONS, "index_type": index_type, **config})
    await store.initialize()
    await store.add_vectors(
        [_row(i) for i in range(4)],
        ids=["a", "b", "c", "d"],
        metadata=[{"tag": "keep"}, {"tag": "drop"}, {"tag": "keep"}, {"tag": "drop"}],
    )
    return store


async def _ids_found(store: FaissVectorStore, axis: int, k: int = 10) -> list[str]:
    return [ext_id for ext_id, _, _ in await store.search(_row(axis), k=k)]


@pytest.mark.parametrize("index_type", INDEX_TYPES)
class TestARewriteReplacesTheRow:
    async def test_the_new_vector_and_metadata_are_read_back(self, index_type: str) -> None:
        store = await _store(index_type)
        await store.add_vectors([_row(5)], ids=["a"], metadata=[{"tag": "new"}])

        [(vector, metadata)] = await store.get_vectors(["a"])
        assert vector is not None
        np.testing.assert_array_equal(vector, _row(5))
        assert metadata == {"tag": "new"}

    async def test_the_old_vector_is_no_longer_searched(self, index_type: str) -> None:
        store = await _store(index_type)
        await store.add_vectors([_row(5)], ids=["a"], metadata=[{"tag": "new"}])

        assert await store.count() == 4
        found = await _ids_found(store, axis=0)
        assert sorted(found) == ["a", "b", "c", "d"]
        assert (await _ids_found(store, axis=5, k=1)) == ["a"]


@pytest.mark.parametrize("index_type", INDEX_TYPES)
class TestADeleteRemovesTheRowEverywhere:
    async def test_every_reader_agrees_it_is_gone(self, index_type: str) -> None:
        store = await _store(index_type)
        assert await store.delete_vectors(["a"]) == 1

        assert await store.get_vectors(["a"]) == [(None, None)]
        assert await store.count() == 3
        found = await _ids_found(store, axis=0)
        assert sorted(found) == ["b", "c", "d"]

    async def test_a_filtered_clear_removes_only_its_matches(self, index_type: str) -> None:
        store = await _store(index_type)
        await store.clear(filter={"tag": "drop"})

        assert await store.count() == 2
        assert await store.count(filter={"tag": "drop"}) == 0
        assert sorted(await _ids_found(store, axis=1)) == ["a", "c"]

    async def test_an_id_deleted_then_written_again_is_one_row(self, index_type: str) -> None:
        store = await _store(index_type)
        await store.delete_vectors(["a"])
        await store.add_vectors([_row(6)], ids=["a"], metadata=[{"tag": "again"}])

        assert await store.count() == 4
        assert (await _ids_found(store, axis=6, k=1)) == ["a"]
        assert sorted(await _ids_found(store, axis=0)) == ["a", "b", "c", "d"]

    async def test_a_search_never_names_an_internal_id(self, index_type: str) -> None:
        store = await _store(index_type)
        await store.delete_vectors(["a", "b"])
        await store.add_vectors([_row(0)], ids=["c"])

        assert set(await _ids_found(store, axis=0)) <= {"c", "d"}


@pytest.mark.parametrize("index_type", INDEX_TYPES)
async def test_a_delete_survives_a_save_and_load(index_type: str, tmp_path: Path) -> None:
    path = str(tmp_path / "store")
    store = await _store(index_type, persist_path=path)
    await store.delete_vectors(["a"])
    await store.add_vectors([_row(5)], ids=["b"])
    await store.close()

    reopened = FaissVectorStore(
        {"dimensions": DIMENSIONS, "index_type": index_type, "persist_path": path}
    )
    await reopened.initialize()
    assert await reopened.count() == 3
    assert sorted(await _ids_found(reopened, axis=0)) == ["b", "c", "d"]
    assert (await _ids_found(reopened, axis=5, k=1)) == ["b"]


# An HNSW graph cannot unlink a node, so an evicted one stays in the index as a
# tombstone every search skips, until the dead outnumber the ratio of live rows
# and the graph is rebuilt from the stored vectors.

NEVER_COMPACT = {"index_params": {"tombstone_compaction_ratio": 100.0}}


class TestATombstoneIsSkippedUntilTheGraphIsRebuilt:
    async def test_a_search_skips_a_node_the_graph_still_holds(self) -> None:
        store = await _store("hnsw", **NEVER_COMPACT)
        await store.delete_vectors(["a"])
        await store.add_vectors([_row(5)], ids=["b"])

        assert store.index.ntotal == 5  # "a" and the old "b" are still nodes
        assert await store.count() == 3
        assert sorted(await _ids_found(store, axis=0)) == ["b", "c", "d"]
        assert sorted(await _ids_found(store, axis=1)) == ["b", "c", "d"]

    async def test_the_graph_is_rebuilt_once_the_dead_pass_a_quarter(self) -> None:
        store = FaissVectorStore({"dimensions": DIMENSIONS, "index_type": "hnsw"})
        await store.initialize()
        await store.add_vectors([_row(i) for i in range(8)], ids=[str(i) for i in range(8)])

        await store.delete_vectors(["0"])  # 1 dead against 7 live: kept
        assert store.index.ntotal == 8
        await store.delete_vectors(["1"])  # 2 dead against 6 live: rebuilt
        assert store.index.ntotal == 6
        assert await store.count() == 6
        assert sorted(await _ids_found(store, axis=0)) == [str(i) for i in range(2, 8)]

    async def test_a_flat_index_removes_in_place(self) -> None:
        store = await _store("flat", **NEVER_COMPACT)
        await store.delete_vectors(["a"])
        assert store.index.ntotal == 3

    async def test_tombstones_survive_a_save_and_load(self, tmp_path: Path) -> None:
        path = str(tmp_path / "store")
        store = await _store("hnsw", persist_path=path, **NEVER_COMPACT)
        await store.delete_vectors(["a"])
        await store.close()

        reopened = FaissVectorStore(
            {"dimensions": DIMENSIONS, "index_type": "hnsw", "persist_path": path, **NEVER_COMPACT}
        )
        await reopened.initialize()
        assert reopened.index.ntotal == 4
        assert await reopened.count() == 3
        assert sorted(await _ids_found(reopened, axis=0)) == ["b", "c", "d"]

    async def test_a_node_a_failed_delete_stranded_is_recovered_on_load(
        self, tmp_path: Path
    ) -> None:
        """A store saved after the old failed delete: the node is in the index
        and the row is in none of the maps. Loading reads it as dead.
        """
        path = str(tmp_path / "store")
        store = await _store("hnsw", persist_path=path, **NEVER_COMPACT)
        internal = store.id_map.pop("a")
        store.metadata_store.pop(internal)
        store.timestamps.pop(internal)
        store.vectors.pop(internal)
        await store.close()

        reopened = FaissVectorStore(
            {"dimensions": DIMENSIONS, "index_type": "hnsw", "persist_path": path}
        )
        await reopened.initialize()
        assert await reopened.count() == 3
        assert str(internal) not in await _ids_found(reopened, axis=0)


async def test_a_flat_index_drops_a_stranded_node_on_load(tmp_path: Path) -> None:
    path = str(tmp_path / "store")
    store = await _store("flat", persist_path=path)
    internal = store.id_map.pop("a")
    store.metadata_store.pop(internal)
    await store.close()

    reopened = FaissVectorStore(
        {"dimensions": DIMENSIONS, "index_type": "flat", "persist_path": path}
    )
    await reopened.initialize()
    assert reopened.index.ntotal == 3
    assert sorted(await _ids_found(reopened, axis=0)) == ["b", "c", "d"]


@pytest.mark.parametrize("ratio", [-0.1, "0.25", True, None])
def test_a_compaction_ratio_must_be_a_non_negative_number(ratio: object) -> None:
    with pytest.raises(ValueError, match="tombstone_compaction_ratio"):
        FaissVectorStore(
            {
                "dimensions": DIMENSIONS,
                "index_type": "hnsw",
                "index_params": {"tombstone_compaction_ratio": ratio},
            }
        )

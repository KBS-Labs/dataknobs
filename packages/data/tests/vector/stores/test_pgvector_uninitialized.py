"""``PgVectorStore`` connects on first use, and must keep doing so.

Nine of the store's public operations open with the same two lines::

    if not self._initialized:
        await self.initialize()

so a store that was constructed but never explicitly initialized still
works: the first operation connects. ``close()`` resets ``_initialized``,
so the same two lines reopen a closed store on next use.

That preamble is what makes ``self._pool`` non-``None`` by the time each
method reaches ``self._pool.acquire()`` --- an invariant the type checker
cannot follow from a flag to an attribute, which is why every one of
those ten sites reported ``Item "None" of "Any | None" has no attribute
"acquire"``. Collapsing the nine copies into one accessor answers the
checker, and these cells are here because doing so could silently answer
it the *wrong* way: an accessor that raised on a ``None`` pool instead of
initializing would satisfy mypy, pass every test that uses an initialized
store, and turn lazy connection into a hard failure for everyone relying
on it.

So what is pinned here is not an error message --- it is that an unopened
store still tries to connect. No PostgreSQL is needed, and none may be
running: the DSN names port 1, which is privileged and bound by nothing,
so the connection is refused immediately. A refusal *is* the evidence,
because reaching it means initialization ran.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("asyncpg")

from dataknobs_data.vector.stores.pgvector import PgVectorStore

# Port 1 is privileged and nothing listens on it, so a connection attempt
# fails immediately rather than hanging or reaching a real server.
UNREACHABLE_DSN = "postgresql://nobody:nothing@127.0.0.1:1/nothing"


def _unopened_store() -> PgVectorStore:
    """A configured store on which ``initialize()`` has never been called."""
    store = PgVectorStore({"connection_string": UNREACHABLE_DSN, "dimensions": 8})
    assert store._pool is None, "construction must not open a pool"
    assert not store._initialized, "construction must not mark the store initialized"
    return store


# The nine operations carrying the lazy-initialize preamble. ``create_index``
# is deliberately absent: it is the tenth pool site and the one that refuses
# rather than initializing, which the last cell pins separately.
LAZY_OPERATIONS: list[tuple[str, Any]] = [
    ("add_vectors", lambda s: s.add_vectors([np.zeros(8, dtype=np.float32)])),
    ("get_vectors", lambda s: s.get_vectors(["a"])),
    ("delete_vectors", lambda s: s.delete_vectors(["a"])),
    ("search", lambda s: s.search(np.zeros(8, dtype=np.float32), k=1)),
    ("update_metadata", lambda s: s.update_metadata(["a"], [{"k": "v"}])),
    ("update_metadata_where", lambda s: s.update_metadata_where(None, {"k": "v"})),
    ("count", lambda s: s.count()),
    ("metadata_fields", lambda s: s.metadata_fields()),
    ("clear", lambda s: s.clear()),
]


class TestFirstUseStillConnects:
    """Lazy initialization survives the pool-accessor refactor."""

    @pytest.mark.parametrize("name,call", LAZY_OPERATIONS, ids=[n for n, _ in LAZY_OPERATIONS])
    @pytest.mark.asyncio
    async def test_operation_on_an_unopened_store_attempts_to_connect(
        self, name: str, call: Any
    ) -> None:
        store = _unopened_store()

        with pytest.raises(OSError) as excinfo:
            await call(store)

        # A connection error proves initialization ran. An AttributeError
        # about NoneType would prove the opposite -- that the operation
        # reached the pool without opening one.
        assert not isinstance(excinfo.value, AttributeError), (
            f"{name} reached the pool without initializing: {excinfo.value!r}"
        )
        assert "NoneType" not in str(excinfo.value), (
            f"{name} dereferenced an unopened pool: {excinfo.value!r}"
        )

    @pytest.mark.asyncio
    async def test_create_index_still_refuses_rather_than_connecting(self) -> None:
        """The tenth pool site guards without initializing, and stays that way.

        ``create_index`` is the one operation that treats an unopened
        store as a caller error rather than a cue to connect. Pinning its
        message keeps the accessor refactor from quietly converting it to
        the lazy shape the other nine use.
        """
        store = _unopened_store()

        with pytest.raises(RuntimeError, match="Store must be initialized before creating index"):
            await store.create_index("hnsw")

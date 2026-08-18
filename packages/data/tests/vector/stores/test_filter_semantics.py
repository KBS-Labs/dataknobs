"""Cross-backend metadata-filter semantics tests.

The four-quadrant semantics are:

* ``scalar`` filter, ``scalar`` metadata — equality.
* ``scalar`` filter, ``list`` metadata — contains (the scalar appears in
  the stored list).
* ``list`` filter, ``scalar`` metadata — IN (the scalar is one of the
  filter elements).
* ``list`` filter, ``list`` metadata — non-empty intersection.

A missing metadata key fails the filter; an empty filter matches every
record. All filter keys must match (AND across keys). Behavior is the
same on every shipping backend (``MemoryVectorStore``,
``FaissVectorStore``, ``ChromaVectorStore``, ``PgVectorStore``).

PgVector additionally must preserve metadata types (booleans stay
booleans, numbers stay numbers) — covered by a small set of
pgvector-only cases.

Result *count* is part of the contract too, not only the matching id
set: a filtered ``search`` returns ``k`` rows whenever ``k`` rows match,
however far outside the unfiltered top-``k`` they fall. The
four-quadrant cases below cannot see that — they run a 5-row corpus at
``k=10`` — so the last section runs a corpus larger than ``k`` on every
backend.
"""

from __future__ import annotations

import contextlib
import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
from dataknobs_common.testing import (
    is_chromadb_available,
    is_faiss_available,
    is_package_available,
    requires_real_postgres,
)

from dataknobs_data.vector.exceptions import VectorDomainScopeError
from dataknobs_data.vector.stores.common import POST_FILTER_OVERFETCH
from dataknobs_data.vector.stores.memory import MemoryVectorStore

if is_faiss_available():
    from dataknobs_data.vector.stores.faiss import FaissVectorStore

if is_chromadb_available():
    from dataknobs_data.vector.stores.chroma import ChromaVectorStore

if is_package_available("asyncpg"):
    from dataknobs_data.vector.stores.pgvector import PgVectorStore


logger = logging.getLogger(__name__)


# requires_real_postgres is exactly the three terms this list assembled by
# hand: a reachable server, TEST_POSTGRES=true, and asyncpg installed.
_pgvector_marks = [requires_real_postgres]


@pytest.fixture
def pgvector_config(make_pgvector_test_table: Any) -> Iterator[dict[str, Any]]:
    """Per-test pgvector config from the shared ``dataknobs-common``
    fixture (pre-drop + teardown drop + pgvector-extension ensure live
    there now). ``metric`` is preserved at ``cosine`` to keep behavior
    byte-identical to the prior hand-rolled config.
    """
    gen = make_pgvector_test_table("test_filter_", dimensions=4)
    cfg = next(gen)
    cfg["metric"] = "cosine"
    try:
        yield cfg
    finally:
        gen.close()


async def _teardown_backend(backend: str, store: Any) -> None:
    """Drop the per-test Chroma collection created by a fixture.

    pgvector tables are owned by the shared ``make_pgvector_test_table``
    fixture (pre-drop + teardown drop), so only Chroma needs explicit
    cleanup here. The Chroma failure is logged rather than swallowed so
    an orphaned test collection becomes visible in pytest output.
    """
    if backend == "chroma":
        try:
            store.client.delete_collection(name=store.collection_name)
        except Exception as exc:
            logger.warning(
                "Chroma teardown failed for collection %r: %s",
                store.collection_name,
                exc,
            )


# Five seed records exercising every metadata shape used by the
# four-quadrant matrix below.
SEED_IDS = ["A", "B", "C", "D", "E"]
SEED_METADATA: list[dict[str, Any]] = [
    {"type": "tension", "tags": ["urgent", "blocker"]},
    {"type": "gap", "tags": ["urgent"]},
    {"type": "tension", "tags": ["later"]},
    {"type": "gap", "tags": []},
    {"type": "terminology"},  # no "tags" key
]


def _seed_vectors() -> np.ndarray:
    """Five 4-d unit vectors (rows of identity, padded). Deterministic."""
    return np.eye(5, 4, dtype=np.float32)


def _query_vector() -> np.ndarray:
    """Constant query vector — distance ordering not asserted."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


# (filter, expected ids) — applied to all four backends.
FOUR_QUADRANT_CASES: list[tuple[dict[str, Any], set[str]]] = [
    ({"type": "tension"}, {"A", "C"}),  # scalar/scalar EQ
    ({"tags": "urgent"}, {"A", "B"}),  # scalar/list contains (NEW)
    ({"type": ["tension", "gap"]}, {"A", "B", "C", "D"}),  # list/scalar IN
    ({"tags": ["urgent", "later"]}, {"A", "B", "C"}),  # list/list intersect
    ({"tags": "missing"}, set()),  # scalar not in list
    ({"type": "tension", "tags": "urgent"}, {"A"}),  # AND across keys
    ({"missing_key": "value"}, set()),  # missing key fails
    ({"tags": []}, set()),  # empty-list filter never satisfied
]

CASE_IDS = [f"case{i + 1}" for i in range(len(FOUR_QUADRANT_CASES))]


@pytest_asyncio.fixture(
    params=[
        pytest.param("memory", id="memory"),
        pytest.param(
            "faiss",
            id="faiss",
            marks=pytest.mark.skipif(not is_faiss_available(), reason="faiss not installed"),
        ),
        pytest.param(
            "chroma",
            id="chroma",
            marks=pytest.mark.skipif(not is_chromadb_available(), reason="chromadb not installed"),
        ),
        pytest.param("pgvector", id="pgvector", marks=_pgvector_marks),
    ]
)
async def any_vector_store(
    request: pytest.FixtureRequest, pgvector_config: dict[str, Any]
) -> AsyncIterator[Any]:
    """Yield a freshly-seeded VectorStore for each backend param."""
    backend = request.param
    store: Any
    if backend == "memory":
        store = MemoryVectorStore({"dimensions": 4})
    elif backend == "faiss":
        store = FaissVectorStore({"dimensions": 4, "metric": "cosine"})
    elif backend == "chroma":
        store = ChromaVectorStore(
            {
                "dimensions": 4,
                "collection_name": f"test_filter_{uuid.uuid4().hex[:8]}",
            }
        )
    elif backend == "pgvector":
        store = PgVectorStore(pgvector_config)
    else:
        pytest.fail(f"Unknown backend param: {backend}")

    await store.initialize()
    try:
        await store.add_vectors(_seed_vectors(), ids=list(SEED_IDS), metadata=list(SEED_METADATA))
        yield store
    finally:
        await _teardown_backend(backend, store)
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("filter_dict,expected", FOUR_QUADRANT_CASES, ids=CASE_IDS)
async def test_search_filter_quadrants(
    any_vector_store: Any,
    filter_dict: dict[str, Any],
    expected: set[str],
) -> None:
    """search() returns the four-quadrant-correct id set for each filter."""
    results = await any_vector_store.search(_query_vector(), k=10, filter=filter_dict)
    assert {r[0] for r in results} == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("filter_dict,expected", FOUR_QUADRANT_CASES, ids=CASE_IDS)
async def test_count_filter_quadrants(
    any_vector_store: Any,
    filter_dict: dict[str, Any],
    expected: set[str],
) -> None:
    """count(filter=...) matches the search result-set size on every backend."""
    n = await any_vector_store.count(filter=filter_dict)
    assert n == len(expected)


# ---------------------------------------------------------------------------
# PgVector type-safety cases (booleans/numerics stop silently returning empty)
#
# These also run under Memory/FAISS/Chroma as a sanity check — Python ``==``
# between like-typed values matches trivially on those backends, and the
# previously broken pgvector text-cast path now matches them too.
# ---------------------------------------------------------------------------


TYPE_SAFETY_IDS = ["X1", "X2"]
TYPE_SAFETY_METADATA: list[dict[str, Any]] = [
    {"active": True, "count": 5},
    {"active": False, "count": 7},
]


@pytest_asyncio.fixture(
    params=[
        pytest.param("memory", id="memory"),
        pytest.param(
            "faiss",
            id="faiss",
            marks=pytest.mark.skipif(not is_faiss_available(), reason="faiss not installed"),
        ),
        pytest.param(
            "chroma",
            id="chroma",
            marks=pytest.mark.skipif(not is_chromadb_available(), reason="chromadb not installed"),
        ),
        pytest.param("pgvector", id="pgvector", marks=_pgvector_marks),
    ]
)
async def type_safety_store(
    request: pytest.FixtureRequest, pgvector_config: dict[str, Any]
) -> AsyncIterator[Any]:
    """Two-record store for type-roundtrip cases."""
    backend = request.param
    store: Any
    if backend == "memory":
        store = MemoryVectorStore({"dimensions": 4})
    elif backend == "faiss":
        store = FaissVectorStore({"dimensions": 4, "metric": "cosine"})
    elif backend == "chroma":
        store = ChromaVectorStore(
            {
                "dimensions": 4,
                "collection_name": f"test_typesafe_{uuid.uuid4().hex[:8]}",
            }
        )
    elif backend == "pgvector":
        store = PgVectorStore(pgvector_config)
    else:
        pytest.fail(f"Unknown backend param: {backend}")

    await store.initialize()
    try:
        vectors = np.eye(2, 4, dtype=np.float32)
        await store.add_vectors(
            vectors,
            ids=list(TYPE_SAFETY_IDS),
            metadata=list(TYPE_SAFETY_METADATA),
        )
        yield store
    finally:
        await _teardown_backend(backend, store)
        await store.close()


@pytest.mark.asyncio
async def test_boolean_true_roundtrip(type_safety_store: Any) -> None:
    n = await type_safety_store.count(filter={"active": True})
    assert n == 1


@pytest.mark.asyncio
async def test_boolean_false_roundtrip(type_safety_store: Any) -> None:
    n = await type_safety_store.count(filter={"active": False})
    assert n == 1


@pytest.mark.asyncio
async def test_numeric_int_roundtrip(type_safety_store: Any) -> None:
    n = await type_safety_store.count(filter={"count": 5})
    assert n == 1


@pytest.mark.asyncio
async def test_numeric_no_implicit_string_coercion(
    type_safety_store: Any,
) -> None:
    """Filter ``{"count": "5"}`` against integer metadata never matches.

    Pre-fix pgvector's text-cast translation also returned 0 here (for
    the wrong reason — both sides stringified). Post-fix it returns 0
    because JSONB ``@>`` is type-preserving and ``"5"`` (string) does
    not contain ``5`` (number).
    """
    n = await type_safety_store.count(filter={"count": "5"})
    assert n == 0


# ---------------------------------------------------------------------------
# Config-level ``domain_id`` scoping.
#
# ``PgVectorStore`` honors a config-level ``domain_id``: every
# read/count/clear/update_metadata_where is implicitly scoped to that
# domain, and ``add_vectors`` defaults a row's ``domain_id`` to it.
# Memory/FAISS/Chroma historically ignored config ``domain_id``
# entirely — the multi-tenant isolation a consumer configures was a
# silent no-op on those backends, and a runtime backend swap changed
# isolation semantics. These reproduce-first tests pin the fixed
# symmetric contract; they fail on memory/faiss/chroma pre-fix and
# pass on pgvector (the expected asymmetry split).
# ---------------------------------------------------------------------------

_DOMAIN_SCOPED_IDS = ["s1", "s2", "o1"]


def _domain_scoped_metadata() -> list[dict[str, Any]]:
    """s1/s2 carry NO ``domain_id`` (must default to config ``t1``);
    o1 explicitly belongs to ``t2`` (must be scoped out).
    """
    return [
        {"k": "v"},
        {"k": "v"},
        # ``secret`` exists only outside the configured domain, so a
        # scoped ``metadata_fields()`` must not report it.
        {"domain_id": "t2", "k": "v", "secret": 1},
    ]


@pytest_asyncio.fixture(
    params=[
        pytest.param("memory", id="memory"),
        pytest.param(
            "faiss",
            id="faiss",
            marks=pytest.mark.skipif(not is_faiss_available(), reason="faiss not installed"),
        ),
        pytest.param(
            "chroma",
            id="chroma",
            marks=pytest.mark.skipif(not is_chromadb_available(), reason="chromadb not installed"),
        ),
        pytest.param("pgvector", id="pgvector", marks=_pgvector_marks),
    ]
)
async def domain_scoped_store(
    request: pytest.FixtureRequest, pgvector_config: dict[str, Any]
) -> AsyncIterator[Any]:
    """Store configured with ``domain_id="t1"``, seeded across t1/t2."""
    backend = request.param
    store: Any
    if backend == "memory":
        store = MemoryVectorStore({"dimensions": 4, "domain_id": "t1"})
    elif backend == "faiss":
        store = FaissVectorStore({"dimensions": 4, "metric": "cosine", "domain_id": "t1"})
    elif backend == "chroma":
        store = ChromaVectorStore(
            {
                "dimensions": 4,
                "domain_id": "t1",
                "collection_name": f"test_domain_{uuid.uuid4().hex[:8]}",
            }
        )
    elif backend == "pgvector":
        store = PgVectorStore({**pgvector_config, "domain_id": "t1"})
    else:
        pytest.fail(f"Unknown backend param: {backend}")

    await store.initialize()
    try:
        await store.add_vectors(
            _seed_vectors()[:3],
            ids=list(_DOMAIN_SCOPED_IDS),
            metadata=_domain_scoped_metadata(),
        )
        yield store
    finally:
        await _teardown_backend(backend, store)
        await store.close()


# NOTE on the asserted contract. A configured ``domain_id`` delivers
# *isolation* symmetry: it confines every read/count/clear/update to
# that domain on every backend, and a cross-domain request is empty.
# That holds however the row is addressed — the filter-keyed surfaces
# get it from ``_effective_filter``, the id-keyed ones
# (``get_vectors``, ``delete_vectors``, ``update_metadata``) and
# ``metadata_fields`` from ``_in_configured_domain``, and pgvector from
# a column predicate. A scope that bound only the surfaces taking a
# filter would be a property of how the caller asks rather than of the
# store.
# The behavior of a caller *explicitly* passing ``domain_id`` in the
# filter is intentionally NOT asserted as uniform: pgvector scopes via
# a dedicated ``domain_id`` column and stores caller metadata JSONB
# verbatim, so an explicit ``{"domain_id": "t1"}`` filter is a
# JSONB-containment probe there, orthogonal to the column scope —
# whereas memory/faiss/chroma carry ``domain_id`` in metadata. That
# divergence is inherent to pgvector's richer schema and is documented
# in VECTOR_FILTER_SEMANTICS.md, not pinned here.


@pytest.mark.asyncio
async def test_config_domain_id_scopes_count(
    domain_scoped_store: Any,
) -> None:
    """count() is implicitly scoped to the configured domain; a
    cross-domain probe intersects to empty on every backend.
    """
    # s1/s2 defaulted to t1; o1 is t2 and scoped out.
    assert await domain_scoped_store.count() == 2
    # Caller asking for a different domain than the configured scope
    # intersects to empty (pgvector: column='t1' AND JSONB-probe 't2';
    # memory/faiss/chroma: AND-merged unsatisfiable filter).
    assert await domain_scoped_store.count(filter={"domain_id": "t2"}) == 0


@pytest.mark.asyncio
async def test_config_domain_id_scopes_search(
    domain_scoped_store: Any,
) -> None:
    """search() never returns rows outside the configured domain."""
    results = await domain_scoped_store.search(_query_vector(), k=10)
    assert {r[0] for r in results} == {"s1", "s2"}
    # Cross-domain request → empty on every backend.
    cross = await domain_scoped_store.search(_query_vector(), k=10, filter={"domain_id": "t2"})
    assert cross == []


@pytest.mark.asyncio
async def test_config_domain_id_scopes_update_metadata_where(
    domain_scoped_store: Any,
) -> None:
    """update_metadata_where(None, ...) only touches the configured
    domain — the count of affected rows is exactly the in-domain set,
    and a cross-domain request is a no-op.
    """
    affected = await domain_scoped_store.update_metadata_where(None, {"_stale": True})
    assert affected == 2
    # An explicit cross-domain update never escapes the configured
    # scope (intersects to empty on every backend).
    cross = await domain_scoped_store.update_metadata_where({"domain_id": "t2"}, {"_stale": True})
    assert cross == 0
    # The scoped store still sees exactly its two in-domain rows.
    assert await domain_scoped_store.count() == 2


@pytest.mark.asyncio
async def test_update_metadata_keeps_the_row_inside_the_configured_domain(
    domain_scoped_store: Any,
) -> None:
    """A replacement dict omitting ``domain_id`` must not unscope the row.

    ``update_metadata`` replaces a row's metadata outright, and on the
    three backends that carry ``domain_id`` *in* that metadata the
    configured scope is one of the keys being replaced. A caller
    updating an unrelated field has no reason to restate it, so the
    write-path default that ``add_vectors`` applies has to apply here
    too — otherwise the row survives the update but leaves the domain.

    pgvector cannot fail this: its ``domain_id`` is a column the
    metadata write never touches. It is in the parametrization as the
    reference the other three have to match.
    """
    assert await domain_scoped_store.update_metadata(["s1"], [{"k": "w"}]) == 1

    # Every scoped surface must still see the row it just updated.
    assert await domain_scoped_store.count() == 2
    assert {r[0] for r in await domain_scoped_store.search(_query_vector(), k=10)} == {"s1", "s2"}
    assert await domain_scoped_store.update_metadata_where(None, {"_swept": True}) == 2


@pytest.mark.asyncio
async def test_update_metadata_leaves_the_row_deletable(
    domain_scoped_store: Any,
) -> None:
    """The unscoped row is not merely invisible — it is unreachable.

    A scoped ``clear()`` resolves to ``{"domain_id": <configured>}`` and
    takes the filtered path, and an absent key never matches a filter.
    So a row that lost its ``domain_id`` cannot be deleted by the store
    that wrote it, while an unscoped store over the same backing data
    still returns it: a leak that outlives the only API that could
    clean it up.
    """
    await domain_scoped_store.update_metadata(["s1"], [{"k": "w"}])
    await domain_scoped_store.clear()

    # Nothing of the configured domain is left behind to be orphaned.
    assert await domain_scoped_store.count() == 0
    # ``get_vectors`` answers positionally, so a row that is really gone
    # comes back as a ``(None, None)`` placeholder rather than being
    # omitted — which is what distinguishes "deleted" from "still there
    # but no longer visible to the scoped surfaces".
    assert await domain_scoped_store.get_vectors(["s1"]) == [(None, None)]


@pytest.mark.asyncio
async def test_get_vectors_does_not_reach_outside_the_configured_domain(
    domain_scoped_store: Any,
) -> None:
    """Knowing an id is not authority to read it.

    ``get_vectors`` is id-keyed, so it never passed through
    ``_effective_filter`` and answered from the whole collection. That
    makes the configured scope a property of *how you ask* rather than
    of the store, which is the opposite of an isolation boundary — and
    ids are frequently derived from content, so they are guessable.
    """
    assert await domain_scoped_store.get_vectors(["o1"]) == [(None, None)]
    # In-domain ids still answer, and position is still preserved.
    rows = await domain_scoped_store.get_vectors(["s1", "o1", "s2"])
    assert [r[0] is not None for r in rows] == [True, False, True]


@pytest.mark.asyncio
async def test_delete_vectors_does_not_reach_outside_the_configured_domain(
    domain_scoped_store: Any,
) -> None:
    """A scoped store cannot delete another domain's row."""
    assert await domain_scoped_store.delete_vectors(["o1"]) == 0
    # Still there: proven from a surface that can see across domains
    # rather than from the scoped count, which would read 2 either way.
    assert await domain_scoped_store.delete_vectors(["s1", "o1"]) == 1


@pytest.mark.asyncio
async def test_update_metadata_does_not_reach_outside_the_configured_domain(
    domain_scoped_store: Any,
) -> None:
    """A scoped store cannot rewrite — or capture — another domain's row.

    Sharper than the read case. Because the write path defaults the
    configured ``domain_id`` into the replacement dict, an unscoped
    ``update_metadata`` would not merely edit the out-of-domain row, it
    would relabel it into the caller's own domain: a cross-domain read
    that leaves no trace it happened.
    """
    assert await domain_scoped_store.update_metadata(["o1"], [{"k": "z"}]) == 0
    # The store still sees only its own two rows — o1 was neither
    # edited nor captured.
    assert await domain_scoped_store.count() == 2
    assert await domain_scoped_store.count(filter={"k": "z"}) == 0


@pytest.mark.asyncio
async def test_metadata_fields_does_not_disclose_another_domain(
    domain_scoped_store: Any,
) -> None:
    """Field *names* are data too.

    ``metadata_fields()`` unions the keys of every stored row, so on a
    scoped store it leaked the shape of every other domain's metadata —
    enough to learn what a neighbour records without reading a row.
    """
    fields = await domain_scoped_store.metadata_fields()
    assert "k" in fields
    assert "secret" not in fields


@contextlib.contextmanager
def _unscoped(store: Any) -> Iterator[Any]:
    """The same store, with its configured scope lifted for the block.

    A second store object is not an option for all four backends —
    Memory and FAISS hold their rows in instance state, so a fresh
    instance shares no data with this one. Lifting ``domain_id`` on the
    object under test is the one way to look at the same backing rows
    from outside the scope on every backend, which is what proving "the
    victim row is still there, still owned by t2" requires.
    """
    saved = store.domain_id
    store.domain_id = None
    try:
        yield store
    finally:
        store.domain_id = saved


@pytest.mark.asyncio
async def test_add_vectors_does_not_capture_another_domain_s_row(
    domain_scoped_store: Any,
) -> None:
    """A scoped store cannot take a row by writing its id.

    The destructive half of the id-keyed hole the read verbs closed.
    ``add_vectors`` upserts on id conflict and the row it writes carries
    the configured scope, so an unguarded write to an id another domain
    owns does not insert alongside it and does not merely edit it — it
    destroys the original and relabels the replacement into the writer's
    own domain. The victim's ``count()`` drops by one and nothing
    anywhere records that it happened.

    pgvector states the capture in its own SQL: the ``ON CONFLICT``
    clause assigns ``domain_id`` from the incoming row.

    Refusing is the only answer that is neither a capture nor a silent
    drop. Ids here are shared across domains by construction — they are
    routinely derived from content — so a collision is a real event
    rather than caller error, and returning ids that were not written
    would be worse than raising.
    """
    with pytest.raises(VectorDomainScopeError) as excinfo:
        await domain_scoped_store.add_vectors(
            _seed_vectors()[:1], ids=["o1"], metadata=[{"k": "mine"}]
        )
    assert "o1" in str(excinfo.value)

    # The victim row is untouched: still owned by t2, still carrying its
    # own metadata rather than the caller's.
    with _unscoped(domain_scoped_store) as store:
        meta = (await store.get_vectors(["o1"]))[0][1] or {}
        assert meta.get("k") == "v"
        assert meta.get("secret") == 1
        assert await store.count() == 3


@pytest.mark.asyncio
async def test_a_rejected_batch_writes_nothing(
    domain_scoped_store: Any,
) -> None:
    """One out-of-domain id rejects the whole batch, before any write.

    Memory, FAISS and Chroma have no transaction to roll back, so a
    guard applied per row as it is written would leave the rows before
    the offending one committed. The check therefore runs over the whole
    batch first: a caller who catches the error and retries is not
    retrying on top of a half-applied write.
    """
    with pytest.raises(VectorDomainScopeError):
        await domain_scoped_store.add_vectors(
            _seed_vectors()[:2],
            ids=["fresh", "o1"],
            metadata=[{"k": "new"}, {"k": "mine"}],
        )

    # ``fresh`` precedes the rejected id in the batch and must not exist.
    assert await domain_scoped_store.get_vectors(["fresh"]) == [(None, None)]
    assert await domain_scoped_store.count() == 2


@pytest.mark.asyncio
async def test_a_scoped_store_still_writes_its_own_ids(
    domain_scoped_store: Any,
) -> None:
    """The guard refuses foreign ids only — in-domain writes are unchanged.

    Both halves matter: re-writing an id the store already owns is an
    ordinary upsert, and a brand-new id is a genuine insert that no
    stored row can object to.
    """
    assert await domain_scoped_store.add_vectors(
        _seed_vectors()[:1], ids=["s1"], metadata=[{"k": "revised"}]
    ) == ["s1"]
    assert await domain_scoped_store.add_vectors(
        _seed_vectors()[:1], ids=["brand-new"], metadata=[{"k": "new"}]
    ) == ["brand-new"]
    assert await domain_scoped_store.count() == 3


@pytest.mark.asyncio
async def test_an_ownerless_row_is_not_the_scoped_store_s_to_take(
    domain_scoped_store: Any,
) -> None:
    """A row with no domain at all is out of scope, not up for grabs.

    Rows written before a scope was configured — or by an unscoped
    admin path — carry no ``domain_id`` (NULL in pgvector's column).
    Every scoped read already treats them as absent, because an absent
    key never satisfies a filter and NULL never equals a value. The
    write side has to agree: silently claiming an ownerless row is the
    same capture as claiming an owned one, just with no victim to
    notice. So the scoped store refuses, and an unscoped store remains
    the way to adopt such rows deliberately.
    """
    with _unscoped(domain_scoped_store) as store:
        await store.add_vectors(_seed_vectors()[:1], ids=["orphan"], metadata=[{"k": "v"}])

    assert await domain_scoped_store.get_vectors(["orphan"]) == [(None, None)]
    with pytest.raises(VectorDomainScopeError):
        await domain_scoped_store.add_vectors(
            _seed_vectors()[:1], ids=["orphan"], metadata=[{"k": "claimed"}]
        )


@pytest.mark.asyncio
async def test_an_unscoped_store_still_writes_any_id(
    domain_scoped_store: Any,
) -> None:
    """The guard is scope-conditional, not a new restriction on ids.

    A store with no configured ``domain_id`` has no scope to violate, so
    every id stays writable — including one carrying another domain's
    tag, which is how a migration or an admin tool addresses the whole
    collection.
    """
    with _unscoped(domain_scoped_store) as store:
        written = await store.add_vectors(
            _seed_vectors()[:1], ids=["o1"], metadata=[{"domain_id": "t2", "k": "rewritten"}]
        )
        assert written == ["o1"]
        assert ((await store.get_vectors(["o1"]))[0][1] or {}).get("k") == "rewritten"


# ---------------------------------------------------------------------------
# A row belonging to more than one domain.
#
# ``domain_id`` is an ordinary metadata key on the three backends that
# carry it in metadata, so the four-quadrant rule at the top of this
# file applies to it like any other: a scalar filter against a list
# value is *membership*. A row tagged ``["t1", "t2"]`` is therefore in
# both domains, and the filter-keyed surfaces have always agreed.
#
# pgvector is absent from the parametrization rather than expected to
# differ: its ``domain_id`` is a scalar column, so the shape cannot be
# stored there at all. That is the same schema divergence the NOTE
# above records for an explicit ``domain_id`` filter.
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(
    params=[
        pytest.param("memory", id="memory"),
        pytest.param(
            "faiss",
            id="faiss",
            marks=pytest.mark.skipif(not is_faiss_available(), reason="faiss not installed"),
        ),
        pytest.param(
            "chroma",
            id="chroma",
            marks=pytest.mark.skipif(not is_chromadb_available(), reason="chromadb not installed"),
        ),
    ]
)
async def multi_domain_store(request: pytest.FixtureRequest) -> AsyncIterator[Any]:
    """Scoped to ``t1``, holding one row that belongs to ``t1`` and ``t2``."""
    backend = request.param
    store: Any
    if backend == "memory":
        store = MemoryVectorStore({"dimensions": 4, "domain_id": "t1"})
    elif backend == "faiss":
        store = FaissVectorStore({"dimensions": 4, "metric": "cosine", "domain_id": "t1"})
    else:
        store = ChromaVectorStore(
            {
                "dimensions": 4,
                "domain_id": "t1",
                "collection_name": f"test_multidomain_{uuid.uuid4().hex[:8]}",
            }
        )

    await store.initialize()
    try:
        await store.add_vectors(
            _seed_vectors()[:1],
            ids=["shared"],
            metadata=[{"domain_id": ["t1", "t2"], "k": "v"}],
        )
        yield store
    finally:
        await _teardown_backend(backend, store)
        await store.close()


@pytest.mark.asyncio
async def test_a_multi_domain_row_is_visible_to_every_scoped_surface(
    multi_domain_store: Any,
) -> None:
    """One scope, one answer — whichever surface asks.

    The filter-keyed surfaces resolve the configured scope through
    ``_match_metadata_filter``, whose scalar-filter/list-metadata
    quadrant is membership. The id-keyed ones went through
    ``_in_configured_domain``, which compared with ``==`` and so read a
    multi-domain row as belonging to neither of its domains.

    The split is the defect, not either answer on its own: ``count()``
    reported the row, ``get_vectors`` reported it absent,
    ``delete_vectors`` refused it, and ``clear()`` removed it anyway —
    so the store disagreed with itself about whether the row existed.
    """
    # Filter-keyed: membership, and this half was always right.
    assert await multi_domain_store.count() == 1
    assert {r[0] for r in await multi_domain_store.search(_query_vector(), k=10)} == {"shared"}

    # Id-keyed: the same scope, so the same answer.
    rows = await multi_domain_store.get_vectors(["shared"])
    assert rows[0][0] is not None
    assert (rows[0][1] or {})["k"] == "v"
    assert "k" in await multi_domain_store.metadata_fields()
    assert await multi_domain_store.update_metadata(["shared"], [{"k": "w"}]) == 1
    assert await multi_domain_store.delete_vectors(["shared"]) == 1
    assert await multi_domain_store.count() == 0


@pytest.mark.asyncio
async def test_mutating_returned_metadata_does_not_rewrite_the_store(
    any_vector_store: Any,
) -> None:
    """A result's metadata dict is the caller's, on every backend.

    Chroma and pgvector build a fresh dict per row on the way out;
    Memory and FAISS returned the stored one, so mutating a search result
    silently rewrote the store on two backends of four. That is both a
    swap-visible difference and a way to corrupt a store without ever
    calling a mutator, so it is pinned here rather than per backend.
    """
    results = await any_vector_store.search(_query_vector(), k=1)
    vector_id, _, metadata = results[0]
    assert metadata is not None

    metadata["injected_by_caller"] = "should not persist"

    again = await any_vector_store.search(_query_vector(), k=1)
    assert "injected_by_caller" not in (again[0][2] or {})
    assert await any_vector_store.count(filter={"injected_by_caller": "should not persist"}) == 0

    # Same contract on the id-keyed read.
    fetched = await any_vector_store.get_vectors([vector_id])
    assert fetched[0][1] is not None
    fetched[0][1]["injected_by_caller"] = "should not persist"
    refetched = await any_vector_store.get_vectors([vector_id])
    assert "injected_by_caller" not in (refetched[0][1] or {})


def _alias_vector() -> np.ndarray:
    """One row to write in the aliasing tests, kept off the seed corpus."""
    return np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)


async def _seed_alias_row(store: Any, vector_id: str, metadata: dict[str, Any]) -> None:
    """Add one row carrying ``metadata``, for an aliasing test to poke at.

    Written per test rather than taken from ``SEED_METADATA``, whose
    dicts are module-level constants — a test that reached a nested value
    through the store would edit them for every test that follows.
    """
    await store.add_vectors(_alias_vector(), ids=[vector_id], metadata=[metadata])


@pytest.mark.asyncio
async def test_mutating_a_nested_value_in_a_result_does_not_rewrite_the_store(
    any_vector_store: Any,
) -> None:
    """The dict handed back is independent at depth, not only at the top.

    Chroma and pgvector reconstruct nested values from JSON on the way
    out, so a list inside a result was already theirs alone. Memory and
    FAISS copied only the outer dict, so ``result["tags"].append(...)``
    still reached the stored row — the same swap-visible difference as
    the test above, one level down, and not covered by it.
    """
    await _seed_alias_row(any_vector_store, "NESTED", {"type": "aliasing", "tags": ["one"]})

    fetched = await any_vector_store.get_vectors(["NESTED"])
    assert fetched[0][1] is not None
    fetched[0][1]["tags"].append("injected by caller")

    refetched = await any_vector_store.get_vectors(["NESTED"])
    assert (refetched[0][1] or {})["tags"] == ["one"]

    # Same contract through the ranked read.
    hits = await any_vector_store.search(_query_vector(), k=5, filter={"type": "aliasing"})
    assert hits and hits[0][2] is not None
    hits[0][2]["tags"].append("injected by caller")

    again = await any_vector_store.search(_query_vector(), k=5, filter={"type": "aliasing"})
    assert (again[0][2] or {})["tags"] == ["one"]


@pytest.mark.asyncio
async def test_mutating_metadata_after_add_vectors_does_not_reach_the_store(
    any_vector_store: Any,
) -> None:
    """A writer keeps no live handle on what it wrote.

    The mirror of the read-side contract: Chroma and pgvector serialize
    on the way in, so the dict the caller passed stopped being the
    store's the moment ``add_vectors`` returned. Memory and FAISS copied
    the outer dict but kept the caller's nested values, so a list the
    caller went on using was the store's list too.
    """
    written: dict[str, Any] = {"type": "inbound_add", "tags": ["one"]}
    await _seed_alias_row(any_vector_store, "INBOUND_ADD", written)

    written["type"] = "rewritten by caller"
    written["tags"].append("injected by caller")

    fetched = await any_vector_store.get_vectors(["INBOUND_ADD"])
    stored = fetched[0][1]
    assert stored is not None
    assert stored["type"] == "inbound_add"
    assert stored["tags"] == ["one"]


@pytest.mark.asyncio
async def test_mutating_metadata_after_update_metadata_does_not_reach_the_store(
    any_vector_store: Any,
) -> None:
    """``update_metadata`` takes a copy too, at every depth.

    Worse than the ``add_vectors`` path on Memory and FAISS, which at
    least copied the outer dict: this one stored the caller's dict
    itself, so even a top-level assignment afterwards rewrote the row.
    """
    await _seed_alias_row(any_vector_store, "INBOUND_UPDATE", {"type": "inbound_update"})

    replacement: dict[str, Any] = {"type": "inbound_update", "tags": ["two"]}
    await any_vector_store.update_metadata(["INBOUND_UPDATE"], [replacement])

    replacement["type"] = "rewritten by caller"
    replacement["tags"].append("injected by caller")

    fetched = await any_vector_store.get_vectors(["INBOUND_UPDATE"])
    stored = fetched[0][1]
    assert stored is not None
    assert stored["type"] == "inbound_update"
    assert stored["tags"] == ["two"]


@pytest.mark.asyncio
async def test_mutating_set_after_update_metadata_where_does_not_reach_the_store(
    any_vector_store: Any,
) -> None:
    """``set_`` is merged as a copy, per row.

    The filter-keyed mutator merges one ``set_`` into every match, so a
    nested value inside it was shared by the caller *and* by every row
    the filter selected — one ``append`` reaching an unbounded number of
    rows at once.
    """
    await _seed_alias_row(any_vector_store, "INBOUND_WHERE_1", {"type": "inbound_where"})
    await _seed_alias_row(any_vector_store, "INBOUND_WHERE_2", {"type": "inbound_where"})

    set_: dict[str, Any] = {"tags": ["three"]}
    assert await any_vector_store.update_metadata_where({"type": "inbound_where"}, set_) == 2

    set_["tags"].append("injected by caller")

    fetched = await any_vector_store.get_vectors(["INBOUND_WHERE_1", "INBOUND_WHERE_2"])
    for _, stored in fetched:
        assert stored is not None
        assert stored["tags"] == ["three"]


# ---------------------------------------------------------------------------
# Result *count* when the filter's matches fall outside the global top-k.
#
# Every case above runs a 5-row corpus at ``k=10``, so ``k >= ntotal``
# always and each backend returns every matching row whatever order it
# searched in. Those cases prove *which* rows match; they cannot prove
# *how many* are returned. A backend that truncates to ``k`` first and
# drops the non-matching rows afterwards passes all of them and still
# returns fewer than ``k`` — frequently zero — as soon as the corpus is
# larger than ``k`` and rows carrying another filter value sit nearer the
# probe.
#
# The decoy count is load-bearing and is not a round number by
# accident. ``"group"`` is not declared in Chroma's
# ``scalar_metadata_keys`` (which defaults to empty), so on that backend
# the filter is a *residual Python post-filter*, not a pushed-down
# ``where`` — and a post-filter is compensated for by over-fetching
# ``k * POST_FILTER_OVERFETCH`` candidates. A corpus at or below that
# window is one Chroma fetches entirely, so its post-filter never
# dilutes and the case cannot fail there however the backend behaves.
#
# That is the same flaw as the one this section exists to fix, one level
# up: a corpus small enough that the truncation under test is
# unreachable. ``_TOPK_DECOYS`` therefore exceeds ``_TOPK_K *
# POST_FILTER_OVERFETCH`` so that every backend has to look past its
# first fetch to answer, and all four legs can fail.
# ---------------------------------------------------------------------------

_TOPK_K = 3
_TOPK_DECOYS = _TOPK_K * POST_FILTER_OVERFETCH + 8
_TOPK_TARGETS = 9


def _topk_corpus() -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """A corpus whose every matching row sits outside the global top-k.

    Row ``i`` is ``[1, t, 0, 0]`` for an increasing ``t``, so similarity
    to the probe ``[1, 0, 0, 0]`` falls monotonically under every metric
    these backends default to. The ``group="other"`` rows take the
    smallest ``t`` and so own the whole leading window — including the
    over-fetched one — and the ``group="target"`` rows follow in order.
    """
    offsets = [0.001 * (i + 1) for i in range(_TOPK_DECOYS)]
    offsets += [0.1 * (i + 1) for i in range(_TOPK_TARGETS)]
    vectors = np.array([[1.0, t, 0.0, 0.0] for t in offsets], dtype=np.float32)
    ids = [f"other{i}" for i in range(_TOPK_DECOYS)]
    ids += [f"target{i}" for i in range(_TOPK_TARGETS)]
    metadata: list[dict[str, Any]] = [{"group": "other"} for _ in range(_TOPK_DECOYS)]
    metadata += [{"group": "target"} for _ in range(_TOPK_TARGETS)]
    return vectors, ids, metadata


def _topk_query() -> np.ndarray:
    """Probe the axis the corpus fans out from."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


@pytest_asyncio.fixture(
    params=[
        pytest.param("memory", id="memory"),
        pytest.param(
            "faiss",
            id="faiss",
            marks=pytest.mark.skipif(not is_faiss_available(), reason="faiss not installed"),
        ),
        pytest.param(
            "chroma",
            id="chroma",
            marks=pytest.mark.skipif(not is_chromadb_available(), reason="chromadb not installed"),
        ),
        pytest.param("pgvector", id="pgvector", marks=_pgvector_marks),
    ]
)
async def topk_store(
    request: pytest.FixtureRequest, pgvector_config: dict[str, Any]
) -> AsyncIterator[Any]:
    """Store seeded with a corpus larger than the ``k`` used below."""
    backend = request.param
    store: Any
    if backend == "memory":
        store = MemoryVectorStore({"dimensions": 4})
    elif backend == "faiss":
        store = FaissVectorStore({"dimensions": 4, "metric": "cosine"})
    elif backend == "chroma":
        store = ChromaVectorStore(
            {
                "dimensions": 4,
                "collection_name": f"test_topk_{uuid.uuid4().hex[:8]}",
            }
        )
    elif backend == "pgvector":
        store = PgVectorStore(pgvector_config)
    else:
        pytest.fail(f"Unknown backend param: {backend}")

    await store.initialize()
    try:
        vectors, ids, metadata = _topk_corpus()
        await store.add_vectors(vectors, ids=ids, metadata=metadata)
        yield store
    finally:
        await _teardown_backend(backend, store)
        await store.close()


@pytest.mark.asyncio
async def test_filtered_search_returns_k_when_k_rows_match(topk_store: Any) -> None:
    """``search`` returns a full ``k`` whenever ``k`` rows match.

    All nine matching rows sit outside the unfiltered top-3, so a
    backend applying the filter after truncating to ``k`` returns
    nothing at all here while ``count`` reports the nine it holds.
    """
    available = await topk_store.count(filter={"group": "target"})
    assert available == _TOPK_TARGETS

    results = await topk_store.search(_topk_query(), k=_TOPK_K, filter={"group": "target"})

    assert len(results) == _TOPK_K, (
        f"store holds {available} matching rows; search(k={_TOPK_K}) returned "
        f"{len(results)}: {[r[0] for r in results]}"
    )
    assert [r[0] for r in results] == ["target0", "target1", "target2"]

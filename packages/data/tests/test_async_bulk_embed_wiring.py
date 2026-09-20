"""``bulk_embed_and_store`` on an async backend actually stores something.

``AsyncBulkEmbedMixin`` existed and was mixed into nothing. All four async
backends inherited the **sync** ``BulkEmbedMixin`` instead, so the method was
not a coroutine function and its ``self.exists`` / ``self.update`` /
``self.create`` calls were made without ``await``.

That fails in the quietest way available. A coroutine object is truthy, so
``if record.id and self.exists(record.id)`` takes the update branch on a record
that does not exist; ``self.update(...)`` returns another coroutine that is
never awaited and never runs; and ``self.create(...)`` returns a coroutine that
is appended to the result list in place of an id. Measured on
``AsyncMemoryDatabase`` before the fix: the call returns
``['coroutine', 'coroutine']`` and the database holds **zero** records. No
exception is raised at any point.

The two mixins were ~100-line near-copies differing only in their ``await``s,
which is why nothing looked wrong at the import site. The shared body now lives
in module-level helpers both call, so the remaining difference between them is
the awaiting --- and a future divergence has nowhere to hide.
"""

from __future__ import annotations

import inspect
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dataknobs_data import Record
from dataknobs_data.backend_selection import (
    KnownBackend,
    available_backends,
    known_backend_classes,
)
from dataknobs_data.backends import async_backends, sync_backends
from dataknobs_data.backends.file import AsyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
from dataknobs_data.fields import VectorField
from dataknobs_data.testing import DeterministicEmbedder
from dataknobs_data.vector.bulk_embed_mixin import AsyncBulkEmbedMixin, BulkEmbedMixin
from dataknobs_data.vector.content import (
    CONTENT_HASH_KEY,
    FIELD_SEPARATOR_KEY,
    SOURCE_FIELDS_KEY,
    compute_content_hash,
)
from dataknobs_data.vector.mixins import AsyncVectorOperationsMixin, SyncVectorOperationsMixin

#: The backends the behavioural half below drives. Three rather than every
#: async backend, because driving one means standing up its store --- which
#: the structural half deliberately does not need, and which is why that half
#: can cover every backend and this one cannot.
BEHAVIOURAL_BACKENDS = ["memory", "file", "sqlite"]

#: Every backend each registry knows of. Derived, because the hand-written
#: version of this list is what row-by-row coverage looks like when nobody
#: updates it: it named seven classes while eleven carry the method, and the
#: four it missed included ``AsyncPostgresDatabase`` --- the async twin of the
#: very class whose stub this guard was written for.
SYNC_BACKENDS = known_backend_classes(sync_backends)
ASYNC_BACKENDS = known_backend_classes(async_backends)


def _embed(texts: list[str]) -> np.ndarray:
    """A deterministic embedding whose first component is the text length."""
    return np.array([[float(len(t)), 1.0, 2.0] for t in texts])


async def _aembed(texts: list[str]) -> np.ndarray:
    """The async form, which only the async mixin can drive."""
    return _embed(texts)


async def _make_async_db(kind: str, root: Path) -> Any:
    if kind == "memory":
        return AsyncMemoryDatabase()
    if kind == "file":
        return AsyncFileDatabase({"path": str(root / "records.json")})
    db = AsyncSQLiteDatabase({"path": str(root / "records.db")})
    await db.connect()
    return db


@pytest.fixture(params=BEHAVIOURAL_BACKENDS)
async def async_db(request: pytest.FixtureRequest) -> AsyncIterator[Any]:
    with tempfile.TemporaryDirectory() as d:
        db = await _make_async_db(request.param, Path(d))
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                await close()


def _reachable(entry: KnownBackend) -> type:
    """The backend's class, or a skip naming the backend that could not be reached.

    A backend importing its driver at module top level is unimportable where
    that driver is absent, and the class is what this guard inspects. Skipping
    is therefore the honest answer --- and it is a skip rather than a silent
    absence from the population, so a run covering five of seven backends says
    which two it did not reach.
    """
    if entry.cls is None:
        pytest.skip(f"{entry.key}: {entry.unavailable_reason}")
    return entry.cls


def _assert_wired(
    entry: KnownBackend,
    *,
    operations: type,
    shared: type,
    awaitable: bool,
) -> None:
    """One backend's ``bulk_embed_and_store``: absent, or the shared one.

    Three facts checked against each other rather than against a list --- what
    the class says it owes, whether it has the method, and where the method
    comes from. Checking the first two against each other is what keeps a
    backend from leaving the population quietly: dropping the operations mixin
    while keeping a hand-rolled body fails here rather than becoming a backend
    this guard has no opinion about.
    """
    cls = _reachable(entry)
    owes = issubclass(cls, operations)
    method = getattr(cls, "bulk_embed_and_store", None)

    if not owes:
        assert method is None, (
            f"{cls.__name__} defines bulk_embed_and_store without inheriting "
            f"{operations.__name__}, so nothing declares what it owes and no "
            f"abstract check covers it. Mix the operations in, or drop the method."
        )
        return

    assert method is not None, (
        f"{cls.__name__} inherits {operations.__name__} without reaching an "
        f"implementation of its abstract bulk_embed_and_store"
    )
    owner = next(k for k in cls.__mro__ if "bulk_embed_and_store" in k.__dict__)
    assert owner is shared, (
        f"{cls.__name__}.bulk_embed_and_store resolves to {owner.__name__}, not "
        f"{shared.__name__}. A backend defining its own body here is either a real "
        f"override worth explaining or a stub standing in for the abstract method"
    )
    expected = "a coroutine function" if awaitable else "a plain def"
    assert inspect.iscoroutinefunction(method) is awaitable, (
        f"{cls.__name__}.bulk_embed_and_store should be {expected} and is not; "
        f"it resolves to {owner.__name__}"
    )


class TestEveryBackendReachesTheSharedImplementation:
    """The structural half, asked of every backend rather than of a list.

    Two wiring failures, one question. ``AsyncBulkEmbedMixin`` existed and was
    mixed into nothing, so every async backend inherited the **sync** body and
    its method was not a coroutine function --- checkable without running
    anything, which is the point: it failed for every async backend while every
    behavioural vector test went on passing, because nothing called this method
    on an async backend. And ``SyncVectorOperationsMixin`` declares
    ``bulk_embed_and_store`` ``@abstractmethod`` while ``BulkEmbedMixin`` is
    what a sync backend is supposed to satisfy it with; a backend that instead
    *stubs* it satisfies ``abc`` just as well --- to ``abc`` a ``raise
    NotImplementedError`` body is an implementation --- so the class constructs,
    the abstract check reports nothing, and the failure waits for the first
    caller. ``SyncPostgresDatabase`` was that backend.

    Both were found one class at a time and fixed one class at a time, and the
    guard written for them named its backends by hand: seven of the eleven
    classes carrying this method. Among the four it did not name was
    ``AsyncPostgresDatabase``, in the same file as the sync class the stub was
    found in. The population is derived now, so a backend added tomorrow is
    covered by having been registered rather than by being remembered here.
    """

    @pytest.mark.parametrize("entry", SYNC_BACKENDS, ids=lambda e: e.key)
    def test_a_sync_backend_resolves_to_the_sync_mixin(self, entry: KnownBackend) -> None:
        _assert_wired(
            entry,
            operations=SyncVectorOperationsMixin,
            shared=BulkEmbedMixin,
            awaitable=False,
        )

    @pytest.mark.parametrize("entry", ASYNC_BACKENDS, ids=lambda e: e.key)
    def test_an_async_backend_resolves_to_the_async_mixin(self, entry: KnownBackend) -> None:
        _assert_wired(
            entry,
            operations=AsyncVectorOperationsMixin,
            shared=AsyncBulkEmbedMixin,
            awaitable=True,
        )


class TestThePopulationIsComplete:
    """A derived list beats a hand-written one only if it is actually complete.

    The failure this replaces was a guard reporting green over four fewer
    backends than it appeared to cover. Deriving the list removes the way that
    happened and introduces another: a derivation that quietly returns less
    than the registry knows fails in exactly the same silence. So the two are
    cross-checked, by an accessor that is not the one under test.
    """

    @pytest.mark.parametrize(
        ("label", "registry", "derived"),
        [
            ("sync", sync_backends, SYNC_BACKENDS),
            ("async", async_backends, ASYNC_BACKENDS),
        ],
        ids=["sync", "async"],
    )
    def test_every_buildable_backend_is_in_it(
        self, label: str, registry: Any, derived: list[KnownBackend]
    ) -> None:
        missing = sorted(set(available_backends(registry)) - {entry.key for entry in derived})

        assert missing == [], f"{label}: {missing} can be built here but is not checked"

    def test_the_two_lanes_know_the_same_backends(self) -> None:
        """Each backend ships as a twin, so a name on one side only is a gap."""
        assert {entry.key for entry in SYNC_BACKENDS} == {entry.key for entry in ASYNC_BACKENDS}

    def test_it_covers_the_four_the_hand_written_list_left_out(self) -> None:
        """The regression guard for the gap itself, named rather than counted.

        A floor, not the population --- these four are in because they were the
        ones missed, and removing one should be a deliberate edit here rather
        than a number quietly going down.
        """
        covered = {
            entry.cls.__name__
            for entry in (*SYNC_BACKENDS, *ASYNC_BACKENDS)
            if entry.cls is not None
        }

        assert {
            "AsyncElasticsearchDatabase",
            "AsyncPostgresDatabase",
            "AsyncS3Database",
            "SyncS3Database",
        } <= covered


class TestTheGuardHasTeeth:
    """What it says to a backend wired each of the three wrong ways.

    An assertion never observed to fail is not evidence that it can, and this
    one now runs over a population nobody maintains --- so the cases it exists
    to catch are constructed here rather than waited for. Each of the three is
    a shape this package has actually shipped.
    """

    def test_a_stub_standing_in_for_the_abstract_method_fails(self) -> None:
        """``SyncPostgresDatabase``, before the stub was deleted."""

        class _Stubbed(BulkEmbedMixin, SyncVectorOperationsMixin):
            def bulk_embed_and_store(self, *args: Any, **kwargs: Any) -> list[str]:
                raise NotImplementedError("placeholder to satisfy the abstract method")

        with pytest.raises(AssertionError, match="resolves to _Stubbed"):
            _assert_wired(
                KnownBackend("probe", _Stubbed, None),
                operations=SyncVectorOperationsMixin,
                shared=BulkEmbedMixin,
                awaitable=False,
            )

    def test_an_async_backend_carrying_the_sync_body_fails(self) -> None:
        """Every async backend, before ``AsyncBulkEmbedMixin`` was mixed into any."""

        class _WrongFlavour(BulkEmbedMixin, AsyncVectorOperationsMixin):
            pass

        with pytest.raises(AssertionError, match="resolves to BulkEmbedMixin"):
            _assert_wired(
                KnownBackend("probe", _WrongFlavour, None),
                operations=AsyncVectorOperationsMixin,
                shared=AsyncBulkEmbedMixin,
                awaitable=True,
            )

    def test_a_method_nothing_declares_fails(self) -> None:
        """The way a backend would leave the population without being noticed."""

        class _Freelance(BulkEmbedMixin):
            pass

        with pytest.raises(AssertionError, match="without inheriting"):
            _assert_wired(
                KnownBackend("probe", _Freelance, None),
                operations=SyncVectorOperationsMixin,
                shared=BulkEmbedMixin,
                awaitable=False,
            )

    def test_an_unreachable_backend_is_skipped_by_name(self) -> None:
        """Not silently dropped: the run says which backend it did not cover."""
        with pytest.raises(pytest.skip.Exception, match="probe: boto3 is not installed"):
            _assert_wired(
                KnownBackend("probe", None, "boto3 is not installed"),
                operations=SyncVectorOperationsMixin,
                shared=BulkEmbedMixin,
                awaitable=False,
            )


class TestTheRecordsAreActuallyStored:
    """The behavioural half: the write lands, and the ids come back."""

    async def test_new_records_are_created(self, async_db: Any) -> None:
        records = [Record(data={"title": "alpha"}), Record(data={"title": "bravo!"})]

        ids = await async_db.bulk_embed_and_store(records, "title", embedding_fn=_embed)

        assert all(isinstance(i, str) for i in ids), f"not ids: {[type(i).__name__ for i in ids]}"
        stored = await async_db.all()
        assert len(stored) == 2, "bulk_embed_and_store stored nothing"

    async def test_the_stored_vector_is_the_embedding(self, async_db: Any) -> None:
        await async_db.bulk_embed_and_store(
            [Record(data={"title": "alpha"})], "title", embedding_fn=_embed
        )

        stored = (await async_db.all())[0]
        field = stored.fields["embedding"]
        # First component is the text length, which pins that the vector came
        # from this record's text rather than from anywhere else.
        assert next(iter(field.value)) == pytest.approx(len("alpha"))

    async def test_an_async_embedding_fn_is_awaited(self, async_db: Any) -> None:
        """The capability the sync mixin cannot provide at all.

        Handed to the sync mixin, an async ``embedding_fn`` returns a coroutine
        that is indexed rather than awaited.
        """
        await async_db.bulk_embed_and_store(
            [Record(data={"title": "alpha"})], "title", embedding_fn=_aembed
        )

        stored = (await async_db.all())[0]
        assert next(iter(stored.fields["embedding"].value)) == pytest.approx(len("alpha"))

    async def test_an_existing_record_is_updated_not_duplicated(self, async_db: Any) -> None:
        """The ``exists`` branch, which the un-awaited call took unconditionally."""
        rid = await async_db.create(Record(data={"title": "alpha"}))
        stored = await async_db.read(rid)
        assert stored is not None

        ids = await async_db.bulk_embed_and_store([stored], "title", embedding_fn=_embed)

        assert ids == [rid]
        assert len(await async_db.all()) == 1, "the update stored a duplicate"


class TestTheDigestSurvivesTheSharedBody:
    """Extracting the shared body must not drop what the sync path records.

    A companion: it passes both before and after, and would fail if the
    refactor lost the content-hash metadata that makes a bulk-embedded vector
    comparable by a synchronizer.
    """

    async def test_metadata_describes_the_assembly(self, async_db: Any) -> None:
        await async_db.bulk_embed_and_store(
            [Record(data={"title": "alpha", "body": "bravo"})],
            ["title", "body"],
            embedding_fn=_embed,
            field_separator=" | ",
        )

        stored = (await async_db.all())[0]
        metadata = stored.fields["embedding"].metadata
        assert metadata[SOURCE_FIELDS_KEY] == ["title", "body"]
        assert metadata[FIELD_SEPARATOR_KEY] == " | "
        assert metadata[CONTENT_HASH_KEY] == compute_content_hash("alpha | bravo")


class TestTheSyncMixinStillWorks:
    """A companion for the extraction: the sync path is unchanged.

    Driven directly rather than through a backend, so it pins the mixin body
    and not a backend's storage conventions.
    """

    def test_sync_bulk_embed_attaches_a_described_vector(self) -> None:
        from dataknobs_data.vector.bulk_embed_mixin import BulkEmbedMixin

        class _Store(BulkEmbedMixin):
            def __init__(self) -> None:
                self.written: dict[str, Record] = {}

            def exists(self, id: str) -> bool:
                return id in self.written

            def update(self, id: str, record: Record) -> bool:
                self.written[id] = record
                return True

            def create(self, record: Record) -> str:
                self.written["minted"] = record
                return "minted"

        store = _Store()
        ids = store.bulk_embed_and_store(
            [Record(data={"title": "alpha"})], "title", embedding_fn=_embed
        )

        assert ids == ["minted"]
        field = store.written["minted"].fields["embedding"]
        assert isinstance(field, VectorField)
        assert field.metadata[CONTENT_HASH_KEY] == compute_content_hash("alpha")


class TestTheEmbedderPath:
    """The typed alternative to ``embedding_fn``, across the same backends.

    ``embedding_fn`` was one of eight incompatible spellings of "turn text
    into vectors" in this package, none of which matched what an LLM provider
    returns. ``embedder=`` is the one shape, and these pin that adopting it
    stores the same thing the callable path stores --- plus the one thing the
    callable path structurally cannot: the identity of the model that produced
    the vectors.
    """

    async def test_an_embedder_stores_records(self, async_db: Any) -> None:
        records = [Record(data={"title": "alpha"}), Record(data={"title": "bravo!"})]

        ids = await async_db.bulk_embed_and_store(
            records, "title", embedder=DeterministicEmbedder(dimensions=8)
        )

        assert all(isinstance(i, str) for i in ids)
        assert len(await async_db.all()) == 2

    async def test_the_stored_vector_is_the_embedders(self, async_db: Any) -> None:
        embedder = DeterministicEmbedder(dimensions=8)

        await async_db.bulk_embed_and_store(
            [Record(data={"title": "alpha"})], "title", embedder=embedder
        )

        stored = (await async_db.all())[0]
        expected = (await embedder.embed(["alpha"]))[0]
        assert list(stored.fields["embedding"].value) == pytest.approx(expected)

    async def test_model_name_defaults_to_the_embedders_identity(self, async_db: Any) -> None:
        """The parameter the caller no longer has to keep in step by hand.

        ``bulk_embed_and_store`` takes ``embedding_fn`` and ``model_name`` as
        independent parameters, so nothing stopped a caller naming one model
        while embedding with another --- and the name is the staleness key, so
        the mismatch is only discovered by a later reader trusting it.
        """
        embedder = DeterministicEmbedder(dimensions=8, model_id="nomic-embed-text")

        await async_db.bulk_embed_and_store(
            [Record(data={"title": "alpha"})], "title", embedder=embedder
        )

        stored = (await async_db.all())[0]
        assert stored.fields["embedding"].model_name == "nomic-embed-text"

    async def test_an_explicit_model_name_still_wins(self, async_db: Any) -> None:
        """Defaulting must not overwrite a caller who said what they meant."""
        await async_db.bulk_embed_and_store(
            [Record(data={"title": "alpha"})],
            "title",
            model_name="caller-said-so",
            embedder=DeterministicEmbedder(dimensions=8),
        )

        stored = (await async_db.all())[0]
        assert stored.fields["embedding"].model_name == "caller-said-so"

    async def test_the_digest_metadata_is_written_either_way(self, async_db: Any) -> None:
        await async_db.bulk_embed_and_store(
            [Record(data={"title": "alpha", "body": "bravo"})],
            ["title", "body"],
            field_separator=" | ",
            embedder=DeterministicEmbedder(dimensions=8),
        )

        metadata = (await async_db.all())[0].fields["embedding"].metadata
        assert metadata[SOURCE_FIELDS_KEY] == ["title", "body"]
        assert metadata[CONTENT_HASH_KEY] == compute_content_hash("alpha | bravo")

    async def test_neither_source_is_an_error_even_with_no_records(self, async_db: Any) -> None:
        """The guard runs before the loop, so an empty input still reports it.

        Routing the choice through ``embed_texts`` inside the loop would make
        this silently return ``[]`` --- the caller told it nothing to embed
        with, and got a successful-looking answer.
        """
        with pytest.raises(ValueError, match="embedder is required"):
            await async_db.bulk_embed_and_store([], "title")

    async def test_both_sources_is_an_error(self, async_db: Any) -> None:
        """One of the two would silently not run, and the caller cannot tell which."""
        with pytest.raises(ValueError, match="not both"):
            await async_db.bulk_embed_and_store(
                [Record(data={"title": "alpha"})],
                "title",
                embedding_fn=_embed,
                embedder=DeterministicEmbedder(dimensions=8),
            )

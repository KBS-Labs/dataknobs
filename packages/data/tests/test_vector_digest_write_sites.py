"""Every path that embeds text into a vector field records a judgeable digest.

``vector/content.py`` documents a vector carrying no ``content_hash`` as
**current**, deliberately: a corpus written before digests existed must not all
re-embed on the first sweep after upgrading. That exemption is safe only while
every *writer* records one, because a writer that does not makes its whole
output permanently exempt from staleness --- and silently, since the sweep that
skips it reports success.

Two writers did not. ``AsyncPostgresDatabase.bulk_embed_and_store`` and
``VectorMigration.run`` each built their own ``VectorField`` instead of the one
the shared helper builds, and each omitted the digest. That is the same defect
the format layer carried at ``72d6a675`` --- "a ``VectorField`` went in and a
plain ``Field`` holding numbers came back, carrying no digest" --- surviving at
the two sites that commit did not reach, and it exists in two places for the
usual reason: the field construction was written out five times.

The postgres half of this needs a live server and is pinned beside its
siblings in ``tests/integration/test_postgres_vector_integration.py``. What is
here is everything reachable in process, plus the census that notices a sixth
copy being written.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Awaitable, Callable

import pytest

from dataknobs_data import Record
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.fields import VectorField
from dataknobs_data.testing import DeterministicEmbedder
from dataknobs_data.vector import (
    CONTENT_HASH_KEY,
    FIELD_SEPARATOR_KEY,
    SOURCE_FIELDS_KEY,
    SyncTextEmbedder,
    compute_content_hash,
)
from dataknobs_data.vector.migration import IncrementalVectorizer, VectorMigration
from dataknobs_data.vector.mixins import VectorSyncMixin
from dataknobs_data.vector.sync import VectorTextSynchronizer

TEXT_FIELDS = ["title", "content"]
SEPARATOR = " | "
ASSEMBLED = "Doc | Body"


def _source() -> Record:
    return Record(data={"title": "Doc", "content": "Body"})


def _embedder() -> DeterministicEmbedder:
    return DeterministicEmbedder(dimensions=8, model_id="v1")


async def _via_bulk_embed_and_store() -> Record:
    db = AsyncMemoryDatabase(config={"vector_enabled": True})
    await db.connect()
    try:
        await db.bulk_embed_and_store(
            [_source()],
            TEXT_FIELDS,
            "embedding",
            embedder=_embedder(),
            field_separator=SEPARATOR,
        )
        [record] = await db.all()
        return record
    finally:
        await db.close()


async def _via_sync_bulk_embed_and_store() -> Record:
    """The synchronous lane, reached the way the seam says to reach it."""
    db = SyncMemoryDatabase(config={"vector_enabled": True})
    db.connect()
    try:
        with SyncTextEmbedder(_embedder()) as bridge:
            db.bulk_embed_and_store(
                [_source()],
                TEXT_FIELDS,
                "embedding",
                embedding_fn=bridge.embed,
                field_separator=SEPARATOR,
            )
        [record] = db.all()
        return record
    finally:
        db.close()


class _TextSyncHost(VectorSyncMixin):
    """The mixin hosted on nothing else.

    As ``test_vector_assembly_consumers`` hosts it: it reads and writes only
    the records it is handed, and no backend in the tree composes it with a
    database.
    """


async def _via_sync_vectors_with_text() -> Record:
    record = _source()
    await _TextSyncHost().sync_vectors_with_text(
        [record],
        TEXT_FIELDS,
        embedder=_embedder(),
        field_separator=SEPARATOR,
    )
    return record


async def _via_synchronizer() -> Record:
    db = AsyncMemoryDatabase(config={"vector_enabled": True})
    await db.connect()
    try:
        await db.create(_source())
        sync = VectorTextSynchronizer(
            database=db,
            embedder=_embedder(),
            text_fields=TEXT_FIELDS,
            field_separator=SEPARATOR,
        )
        await sync.sync_all()
        [record] = await db.all()
        return record
    finally:
        await db.close()


async def _via_migration() -> Record:
    source = AsyncMemoryDatabase(config={"vector_enabled": True})
    await source.connect()
    target = AsyncMemoryDatabase(config={"vector_enabled": True})
    await target.connect()
    try:
        await source.create(_source())
        migration = VectorMigration(
            source_db=source,
            target_db=target,
            embedder=_embedder(),
            text_fields=TEXT_FIELDS,
            field_separator=SEPARATOR,
        )
        status = await migration.run()
        assert status.failed_records == 0, status.errors
        [record] = await target.all()
        return record
    finally:
        await source.close()
        await target.close()


WRITE_PATHS: dict[str, Callable[[], Awaitable[Record]]] = {
    "bulk_embed_and_store": _via_bulk_embed_and_store,
    "bulk_embed_and_store-sync": _via_sync_bulk_embed_and_store,
    "sync_vectors_with_text": _via_sync_vectors_with_text,
    "VectorTextSynchronizer": _via_synchronizer,
    "VectorMigration": _via_migration,
}


@pytest.mark.asyncio
class TestEveryWriterDescribesWhatItEmbedded:
    """One assertion, applied to every in-process path that writes a vector."""

    @pytest.mark.parametrize("path", WRITE_PATHS.values(), ids=list(WRITE_PATHS))
    async def test_the_digest_is_written_over_the_assembly_a_reader_repeats(
        self,
        path: Callable[[], Awaitable[Record]],
    ) -> None:
        """A digest alone is not enough --- it has to describe its own input.

        The digest is compared against text a *reader* reassembles, so a writer
        that records the hash without the fields and separator it was computed
        over leaves the reader guessing. Guessing a space where the writer used
        something else reports every record outdated, permanently, which is the
        failure ``b3493e0b`` measured on ``ChangeTracker``.
        """
        record = await path()

        field = record.fields.get("embedding")
        assert isinstance(field, VectorField)
        metadata = field.metadata or {}

        assert metadata.get(CONTENT_HASH_KEY) == compute_content_hash(ASSEMBLED)
        assert metadata.get(SOURCE_FIELDS_KEY) == TEXT_FIELDS
        assert metadata.get(FIELD_SEPARATOR_KEY) == SEPARATOR


@pytest.mark.asyncio
class TestThePlainValueLane:
    """The third writer, which does not build a ``VectorField`` at all.

    ``IncrementalVectorizer`` stores the vector as a bare list and describes it
    in a ``{field}_metadata`` sidecar. That is a different shape, not a
    different contract: a reader still has to decide whether the stored vector
    still matches its source text, and the answer was that it could not --- the
    digest comparison was written only into the ``VectorField`` lane, and the
    sidecar had nowhere to keep a digest anyway.

    So the same corpus, the same edit, judged by the same synchronizer, came
    out differently depending only on which class had embedded it. The sidecar
    now carries the same three keys ``content_hash_metadata`` writes onto a
    ``VectorField``, and both lanes ask one function about them.
    """

    async def _vectorized(self, db: AsyncMemoryDatabase) -> Record:
        await db.create(_source())
        vectorizer = IncrementalVectorizer(
            database=db,
            embedder=_embedder(),
            text_fields=TEXT_FIELDS,
            field_separator=SEPARATOR,
        )
        [record] = await db.all()
        assert await vectorizer._process_record(record) is True
        [stored] = await db.all()
        return stored

    def _sweep(self, db: AsyncMemoryDatabase) -> VectorTextSynchronizer:
        return VectorTextSynchronizer(
            database=db,
            embedder=_embedder(),
            text_fields=TEXT_FIELDS,
            field_separator=SEPARATOR,
        )

    async def test_the_sidecar_carries_the_digest_and_its_assembly(self) -> None:
        db = AsyncMemoryDatabase(config={"vector_enabled": True})
        await db.connect()
        try:
            stored = await self._vectorized(db)

            # Still a plain value, deliberately: this changes what the sidecar
            # says, not what the vector is.
            assert not isinstance(stored.fields.get("embedding"), VectorField)

            sidecar = stored.get_value("embedding_metadata")
            assert sidecar[CONTENT_HASH_KEY] == compute_content_hash(ASSEMBLED)
            assert sidecar[SOURCE_FIELDS_KEY] == TEXT_FIELDS
            assert sidecar[FIELD_SEPARATOR_KEY] == SEPARATOR
        finally:
            await db.close()

    async def test_an_edited_record_is_re_embedded(self) -> None:
        """Measured before the fix: ``updated=0``, however far the text drifts."""
        db = AsyncMemoryDatabase(config={"vector_enabled": True})
        await db.connect()
        try:
            stored = await self._vectorized(db)
            stored.set_value("content", "Body, substantially rewritten")
            await db.update(stored.id, stored)

            assert (await self._sweep(db).sync_all())["updated"] == 1
        finally:
            await db.close()

    async def test_an_untouched_record_is_left_alone(self) -> None:
        db = AsyncMemoryDatabase(config={"vector_enabled": True})
        await db.connect()
        try:
            await self._vectorized(db)
            assert (await self._sweep(db).sync_all())["updated"] == 0
        finally:
            await db.close()

    async def test_a_sidecar_with_no_digest_is_still_current(self) -> None:
        """The exemption that keeps a pre-digest corpus from re-embedding.

        Spelled out for this lane because the version check directly beside it
        goes the other way: an absent sidecar is a version *mismatch* there.
        The two rules genuinely differ, so neither can be read off the other.
        """
        db = AsyncMemoryDatabase(config={"vector_enabled": True})
        await db.connect()
        try:
            stored = await self._vectorized(db)
            sidecar = dict(stored.get_value("embedding_metadata"))
            del sidecar[CONTENT_HASH_KEY]
            stored.set_value("embedding_metadata", sidecar)
            stored.set_value("content", "Body, substantially rewritten")
            await db.update(stored.id, stored)

            assert (await self._sweep(db).sync_all())["updated"] == 0
        finally:
            await db.close()


@pytest.mark.asyncio
class TestTheConsequenceForAMigratedCorpus:
    """Why the digest matters, stated as the behaviour it buys."""

    async def test_an_edited_record_is_re_embedded_after_a_migration(self) -> None:
        """Without a digest this sweep reports success and does nothing.

        Measured before the fix: ``sync_all`` returns ``updated=0`` over a
        corpus whose source text has been rewritten, because the migrated
        vector carried nothing to judge it by.
        """
        db = AsyncMemoryDatabase(config={"vector_enabled": True})
        await db.connect()
        source = AsyncMemoryDatabase(config={"vector_enabled": True})
        await source.connect()
        try:
            await source.create(_source())
            migration = VectorMigration(
                source_db=source,
                target_db=db,
                embedder=_embedder(),
                text_fields=TEXT_FIELDS,
                field_separator=SEPARATOR,
            )
            await migration.run()

            [record] = await db.all()
            record.set_value("content", "Body, substantially rewritten")
            await db.update(record.id, record)

            sync = VectorTextSynchronizer(
                database=db,
                embedder=_embedder(),
                text_fields=TEXT_FIELDS,
                field_separator=SEPARATOR,
            )
            assert (await sync.sync_all())["updated"] == 1
        finally:
            await source.close()
            await db.close()

    async def test_an_untouched_record_is_left_alone(self) -> None:
        """The half a digest written under the wrong assembly would break.

        A hash computed over one string and compared against another differs
        every time, so a writer that mis-describes its own input is
        indistinguishable from one that omits the digest until you sweep a
        corpus nobody edited.
        """
        source = AsyncMemoryDatabase(config={"vector_enabled": True})
        await source.connect()
        db = AsyncMemoryDatabase(config={"vector_enabled": True})
        await db.connect()
        try:
            await source.create(_source())
            migration = VectorMigration(
                source_db=source,
                target_db=db,
                embedder=_embedder(),
                text_fields=TEXT_FIELDS,
                field_separator=SEPARATOR,
            )
            await migration.run()

            sync = VectorTextSynchronizer(
                database=db,
                embedder=_embedder(),
                text_fields=TEXT_FIELDS,
                field_separator=SEPARATOR,
            )
            assert (await sync.sync_all())["updated"] == 0
        finally:
            await source.close()
            await db.close()


# Every place in the package that constructs a `VectorField`, and what makes it
# exempt from the rule above. Keyed on the enclosing qualified name rather than
# a line number, so ordinary edits do not touch it.
VECTOR_FIELD_CONSTRUCTION_SITES = {
    ("vector/bulk_embed_mixin.py", "attach_vector_field"): (
        "writes the digest; the helper every embedding writer now routes through"
    ),
    ("vector/mixins.py", "VectorSyncMixin.sync_vectors_with_text"): "writes the digest",
    ("vector/sync.py", "VectorTextSynchronizer.sync_record"): "writes the digest",
    ("vector/mixins.py", "vector_field_for"): (
        "`update_vector` takes a caller-supplied vector and metadata, with no "
        "text to digest -- the caller replaced the value and owns what it means"
    ),
    ("backends/sqlite.py", "SyncSQLiteDatabase.add_vectors"): (
        "the raw vector-store API: ids and vectors, no source text anywhere"
    ),
    ("backends/elasticsearch_mixins.py", "ElasticsearchRecordSerializer._document_to_record"): (
        "reads a stored document back and restores the metadata it carried, "
        "which is how a digest survives the round trip rather than being made"
    ),
    ("records.py", "Record.copy"): "deep-copies the metadata it was given",
}


class TestTheCensusOfConstructionSites:
    """A sixth copy of building the field is what put this defect in two places.

    So adding one is a test failure rather than a silence. This cannot tell
    whether a new site *needs* a digest --- that is a judgement, and the
    mapping above is where it gets recorded --- but it does refuse to let the
    judgement go unmade.
    """

    def test_no_construction_site_is_unaccounted_for(self) -> None:
        # Deliberately not `async def`: this reads the source tree and awaits
        # nothing, so the blocking `Path` work has no event loop to stall.
        package = pathlib.Path(__file__).resolve().parents[1] / "src" / "dataknobs_data"
        found: set[tuple[str, str]] = set()

        for path in sorted(package.rglob("*.py")):
            scope: list[str] = []
            relative = path.relative_to(package).as_posix()

            def visit(node: ast.AST, scope: list[str] = scope, relative: str = relative) -> None:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    scope.append(node.name)
                    for child in ast.iter_child_nodes(node):
                        visit(child)
                    scope.pop()
                    return
                if isinstance(node, ast.Call):
                    func = node.func
                    name = getattr(func, "id", None) or getattr(func, "attr", None)
                    if name == "VectorField":
                        found.add((relative, ".".join(scope)))
                for child in ast.iter_child_nodes(node):
                    visit(child)

            visit(ast.parse(path.read_text()))

        assert found == set(VECTOR_FIELD_CONSTRUCTION_SITES), (
            "a VectorField is built somewhere this file does not account for; "
            "if it embeds text it must record a digest -- prefer "
            "`attach_vector_field` -- and either way say so above"
        )

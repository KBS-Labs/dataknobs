"""The ``database`` grounded source must be able to name what it reads.

The source builds its own backend from config. Until now the only key it
forwarded was ``backend``, plus a ``connection`` string that no backend
accepts under any spelling -- so the backend it built was always the
default-configured one, and the only backend needing no configuration is
the in-process store. That is the one configuration under which every
gap below is invisible:

* a ``path`` naming a file is dropped, so a persistent backend silently
  gets ``:memory:``;
* the database is never connected, and a backend that needs connecting
  raises on first query -- which the retrieval loop logs and drops, so a
  source grounded on nothing contributes nothing on every turn;
* ``schema.fields`` written as the documented list of ``{name, type}``
  mappings raises, because the builder only reads the mapping form.

These pin the source against a real file-backed store, which is the
smallest configuration that can tell an empty database from an unreachable
one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from dataknobs_bots.knowledge.sources.factory import _create_database_source
from dataknobs_bots.reasoning.grounded_config import GroundedSourceConfig
from dataknobs_bots.testing import BotTestHarness, GroundedConfigBuilder
from dataknobs_common.exceptions import ValidationError
from dataknobs_data import Record, async_database_factory
from dataknobs_data.fields import FieldType
from dataknobs_data.schema import DatabaseSchema
from dataknobs_data.sources.base import RetrievalIntent

#: The module that decides the backend, and so the only one that can report
#: having guessed one.
SELECTION_LOGGER = "dataknobs_data.backend_selection"

#: ``schema.fields`` exactly as the grounded-reasoning guide writes it: a
#: list of mappings, not a mapping of names. Used only where that form is
#: the subject, so every other case below fails for its own reason.
DOCUMENTED_FIELD_LIST = [
    {"name": "title", "type": "string"},
    {"name": "summary", "type": "text"},
]

#: The mapping form, which the builder has always read.
FIELDS = {"title": "string", "summary": "text"}


async def _populated_store(path: Path) -> None:
    """Write one record into a SQLite file through a connected database.

    Real backend, real file: the source under test has to reach this same
    file to see the record, which is what makes "zero results" and "cannot
    reach the store" distinguishable.
    """
    db = async_database_factory.create(backend="sqlite", path=str(path), table="cases")
    await db.connect()
    db.set_schema(DatabaseSchema.create(title=FieldType.STRING, summary=FieldType.TEXT))
    await db.create(Record({"title": "Widget recall", "summary": "A widget was recalled."}))
    await db.close()


def _config(**options: Any) -> GroundedSourceConfig:
    return GroundedSourceConfig(name="case_studies", source_type="database", options=options)


async def test_the_documented_schema_field_list_builds() -> None:
    """``schema.fields`` as a list of mappings is the documented form."""
    source = await _create_database_source(
        _config(backend="memory", schema={"fields": DOCUMENTED_FIELD_LIST})
    )

    assert set(source.get_schema().fields) >= {"title", "summary"}


async def test_a_backend_option_reaches_the_backend(tmp_path: Path) -> None:
    """A ``path`` naming a store is forwarded, so the source reads that store.

    Fails when the key is dropped: the backend falls back to ``:memory:``
    and the record written to the file is not there to find.
    """
    store = tmp_path / "cases.db"
    await _populated_store(store)

    source = await _create_database_source(
        _config(
            backend="sqlite",
            path=str(store),
            table="cases",
            content_field="summary",
            text_search_fields=["title", "summary"],
            schema={"fields": FIELDS},
        )
    )
    results = await source.query(RetrievalIntent(text_queries=["Widget"]))
    await source.close()

    assert [r.content for r in results] == ["A widget was recalled."]


async def test_a_built_backend_is_connected(tmp_path: Path) -> None:
    """The source connects what it built, so its first query can answer.

    This asserted on a log once. It cannot any longer:
    :class:`DatabaseSource` no longer absorbs a failed query, so it no
    longer reports one either -- ``sources/database.py`` has no logger
    at all -- and an assertion that the source logged nothing would now
    hold whatever the factory did.

    So it asserts on the record instead. Drop the ``connect`` and the
    backend raises ``RuntimeError`` here rather than returning ``[]``,
    which fails this test as an error rather than an assertion. That the
    raise is what a caller now meets is the subject of
    ``packages/data/tests/test_database_source_failure_surfaces.py``;
    what is pinned here is that a source the factory built does not meet
    it.
    """
    store = tmp_path / "cases.db"
    await _populated_store(store)

    source = await _create_database_source(
        _config(backend="sqlite", path=str(store), table="cases", schema={"fields": FIELDS})
    )
    results = await source.query(RetrievalIntent(text_queries=["Widget"]))
    await source.close()

    # One record was seeded, and reaching it at all is the claim here.
    # This config names no ``content_field``, so the content comes back
    # empty; that it is the seeded record is the sibling test's subject.
    assert len(results) == 1


async def test_an_option_no_backend_accepts_is_reported() -> None:
    """``connection`` is not a key any backend takes, under any spelling.

    It was carried in the source's documented option list and forwarded
    verbatim. Pinned here because the message a consumer meets is the only
    thing that tells them which key to write instead.
    """
    with pytest.raises(ValueError, match="connection"):
        await _create_database_source(
            _config(backend="postgres", connection="postgresql://host/db")
        )


async def test_a_misspelled_option_is_reported_rather_than_ignored() -> None:
    """An option matching neither the source nor the backend is a fault.

    Silently dropping it leaves a source configured differently from what
    its config says, which is the condition every case above shares.
    """
    with pytest.raises(ValueError, match="content_fields"):
        await _create_database_source(_config(backend="memory", content_fields="summary"))


async def test_the_source_own_options_are_not_offered_to_the_backend() -> None:
    """The source's keys stay with the source.

    Every backend rejects an unrecognised key now, so forwarding these
    would turn the documented configuration into an error.
    """
    source = await _create_database_source(
        _config(
            backend="memory",
            content_field="summary",
            text_search_fields=["title", "summary", "tags"],
            schema={"fields": FIELDS},
            description="Case studies",
        )
    )
    await source.close()

    assert source.get_schema().description == "Case studies"


async def test_a_config_naming_no_backend_still_reports_the_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Forwarding the rest must not start supplying a backend of its own.

    The factory reports an unnamed backend at WARNING because an empty
    config and one asking for an in-process store are not the same event.
    That report is only reachable while the absence survives the trip.
    """
    with caplog.at_level(logging.DEBUG, logger=SELECTION_LOGGER):
        source = await _create_database_source(_config(schema={"fields": FIELDS}))
    await source.close()

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1


async def test_an_unusable_schema_fields_shape_is_refused() -> None:
    """``schema.fields`` written as neither shape is refused, naming the source.

    A schema silently built with no fields describes a source that can be
    filtered on nothing, which reads downstream as a source with nothing
    to offer -- so it is not built at all.
    """
    with pytest.raises(ValidationError, match="'case_studies'"):
        await _create_database_source(_config(backend="memory", schema={"fields": "title"}))


async def test_a_schema_that_is_not_a_mapping_is_reported() -> None:
    """``schema`` carries a ``fields`` key; anything else names the source."""
    with pytest.raises(ValueError, match="case_studies"):
        await _create_database_source(_config(backend="memory", schema=["title"]))


async def test_a_rejected_config_opens_no_store(tmp_path: Path) -> None:
    """A config rejected for its schema must not have opened a store first.

    The shape of ``schema`` is a property of the config alone, so it is
    knowable before anything is opened. Checking it afterwards is not a
    tidiness question: :meth:`AsyncSQLiteDatabase.connect` creates the
    parent directories, creates the file, and creates the table, and the
    raise then abandons all three along with the connection -- so a
    config that was rejected still leaves a store behind, and the next
    run finds one already there.

    ``memory`` cannot show this. Its ``connect`` is the base no-op, which
    is why the sibling above pins the message and this pins the ordering.
    """
    store = tmp_path / "unwritten" / "cases.db"

    with pytest.raises(ValueError, match="'schema' must be a mapping"):
        await _create_database_source(
            _config(backend="sqlite", path=str(store), table="cases", schema=["title"])
        )

    assert not store.exists()
    assert not store.parent.exists()


# --------------------------------------------------------------------------
# ``schema:`` is read by the shared reader, or refused naming the source
# --------------------------------------------------------------------------


async def _refusal(schema: Any) -> str:
    """The message a ``schema:`` the reader refuses reaches the caller with."""
    with pytest.raises(ValidationError) as excinfo:
        await _create_database_source(_config(backend="memory", schema=schema))
    message = str(excinfo.value)
    assert message.startswith("source 'case_studies': ")
    assert excinfo.value.context["source"] == "case_studies"
    return message


async def test_an_unknown_field_type_is_refused_naming_the_field() -> None:
    """It used to become ``string``, with a warning in one spelling and none in the other."""
    for fields in ({"title": "money"}, {"title": {"type": "money"}}):
        message = await _refusal({"fields": fields})
        assert "'title'" in message and "'money'" in message


async def test_a_field_key_the_reader_does_not_take_is_refused() -> None:
    """``description:`` beside ``type:`` used to be ignored; it lives under ``metadata:``."""
    message = await _refusal({"fields": {"title": {"type": "string", "description": "x"}}})
    assert "'description'" in message


@pytest.mark.parametrize(
    "fields",
    [{"title": None, "summary": "text"}, [{"type": "string"}, {"name": "summary"}]],
    ids=["null-field", "row-with-no-name"],
)
async def test_a_field_that_cannot_be_placed_is_refused(fields: Any) -> None:
    """Both used to be warned about and skipped, building a schema short a column."""
    await _refusal({"fields": fields})


async def test_a_repeated_row_name_is_refused() -> None:
    """The last row used to win without a word."""
    message = await _refusal(
        {"fields": [{"name": "title", "type": "string"}, {"name": "title", "type": "text"}]}
    )
    assert "'title'" in message


async def test_columns_beside_fields_are_refused() -> None:
    """``schema: {title: string}`` used to build a source with no fields, silently."""
    message = await _refusal({"title": "string"})
    assert "['title']" in message and "fields:" in message


async def test_a_non_string_mapping_key_is_refused() -> None:
    """``{5: integer}`` reached ``DatabaseSchema.create(**kwargs)`` as a bare ``TypeError``."""
    message = await _refusal({"fields": {5: "integer"}})
    assert "5" in message


async def test_a_refused_schema_opens_no_store(tmp_path: Path) -> None:
    """The reader runs on the config alone, so before ``connect()`` creates anything.

    The sibling above pins that ordering for bots' own mapping check; this
    pins it for the shared reader's refusals, which arrive from the same spot.
    """
    store = tmp_path / "unwritten" / "cases.db"

    with pytest.raises(ValidationError, match="'money'"):
        await _create_database_source(
            _config(
                backend="sqlite",
                path=str(store),
                table="cases",
                schema={"fields": {"title": "money"}},
            )
        )

    assert not store.exists()
    assert not store.parent.exists()


async def test_a_field_description_becomes_the_filter_description() -> None:
    """``metadata.description`` used to be dropped, leaving ``Filter on <field>``."""
    source = await _create_database_source(
        _config(
            backend="memory",
            schema={
                "fields": {
                    "title": {"type": "string", "metadata": {"description": "The case title"}}
                }
            },
        )
    )
    await source.close()

    assert source.get_schema().fields["title"]["description"] == "The case title"


async def test_enum_still_reaches_the_filter_schema() -> None:
    """The shorthand bots read itself is now the shared reader's, in both spellings."""
    for fields in (
        {"dept": {"type": "string", "enum": ["CS", "Math"]}},
        [{"name": "dept", "type": "string", "enum": ["CS", "Math"]}],
    ):
        source = await _create_database_source(_config(backend="memory", schema={"fields": fields}))
        await source.close()

        dept = source.get_schema().fields["dept"]
        assert dept["enum"] == ["CS", "Math"]
        assert dept["x-extraction"] == {"normalize": True}


async def test_a_type_name_still_reads_in_any_case() -> None:
    """Bots matched type names case-insensitively, and the shared reader now does."""
    source = await _create_database_source(
        _config(
            backend="memory", schema={"fields": {"title": "String", "summary": {"type": "TEXT"}}}
        )
    )
    await source.close()

    fields = source.get_schema().fields
    assert fields["title"]["type"] == fields["summary"]["type"] == "string"


async def test_a_type_with_no_value_is_a_string_in_both_spellings() -> None:
    """YAML's ``type:`` with no value used to be a bare ``AttributeError``."""
    for fields in ({"title": {"type": None}}, [{"name": "title", "type": None}]):
        source = await _create_database_source(_config(backend="memory", schema={"fields": fields}))
        await source.close()

        assert source.get_schema().fields["title"]["type"] == "string"


async def test_a_bot_whose_source_declares_an_unknown_type_is_not_built() -> None:
    """The refusal reaches whoever built the bot, as the reader raised it.

    ``create_source_from_config`` unwraps the registry's ``OperationError``
    so a builder's own error type arrives intact; a bot config can declare
    several sources, so the message names which one.
    """
    config = (
        GroundedConfigBuilder()
        .intent(mode="static", text_queries=["widgets"])
        .synthesis(mode="template", template="{{ results|length }}")
        .source(
            type="database",
            name="case_studies",
            backend="memory",
            schema={"fields": {"title": "money"}},
        )
        .build()
    )

    with pytest.raises(ValidationError, match=r"source 'case_studies'.*'money'"):
        async with await BotTestHarness.create(bot_config=config, main_responses=[]):
            pass

# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The adapter from a vocabulary to an index, and what it refuses at construction.

No vector store, no embedder and no ``dataknobs-data`` import anywhere in this
module. The adapter's refusals and its enumeration are ``dataknobs-common``'s
in full, which is what keeps this package's suite installable with nothing but
its own dependencies --- the property the placement rule exists for.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import (
    ALIAS_FORMS_KEY,
    ONTOLOGY_ID_KEY,
    EntitySourceIndexSource,
    async_load_ontology,
)
from dataknobs_common.ontology.sources import SourceDescription

if TYPE_CHECKING:
    from dataknobs_common.ontology import AsyncOntology

#: A vocabulary declaring three types and holding entities under two of them.
#:
#: The third type is what makes the enumeration assertable: a suite whose
#: every declared type holds something cannot tell *enumerate the schema* from
#: *enumerate what is held*, and those are different answers over a real
#: vocabulary.
THREE_TYPES = """\
ontology:
  id: catalog
  version: "1.0"

  entity_types:
    - id: Product
    - id: Brand
    - id: Discontinued

  entities:
    - {id: sku-4471, type: Product, name: Acme Widget,
       description: a widget, aliases: [Widget, "ACME widget"]}
    - {id: sku-8802, type: Product, name: Bolt}
    - {id: acme,     type: Brand,   name: Acme Corp, aliases: [ACME]}
"""


async def _catalog(tmp_path: Path) -> AsyncOntology[str]:
    path = tmp_path / "catalog.yaml"
    path.write_text(THREE_TYPES)
    return await async_load_ontology(path)


class _Unenumerable:
    """An entity source that cannot say which types it holds.

    Not a mock: it is a real object answering the one member the adapter
    consults at construction, in the one state a live binding over an
    undeclared table is genuinely in. Nothing else is reached before the
    refusal, which is the point of refusing at construction.
    """

    def __init__(self, declares: frozenset[str] | None) -> None:
        self._declares = declares

    def describe(self) -> SourceDescription:
        return SourceDescription("products_table", "memory", None, {}, frozenset(), self._declares)

    async def by_type(self, type_id: str) -> frozenset[str]:
        return frozenset()

    async def get_many(self, entity_ids: Any) -> dict[str, Any]:
        return {}


def _with_source(ontology: AsyncOntology[str], source: Any) -> AsyncOntology[str]:
    """The same vocabulary with a different entity source behind it."""
    import dataclasses

    return dataclasses.replace(ontology, entities=source)


# --------------------------------------------------------------------------
# The two states of `declares`, in one test
# --------------------------------------------------------------------------


async def test_declares_none_is_refused_and_an_empty_set_builds_an_empty_index(
    tmp_path: Path,
) -> None:
    """*Cannot enumerate* is refused; *holds none* is accepted and yields nothing.

    **Both clauses in one test, because the difference between them is the
    contract** rather than two behaviours that happen to coexist. A suite that
    keeps only the refusal has asserted that the adapter is careful; it has
    not asserted that an empty vocabulary is a legitimate answer, which is the
    half that stops an empty index reading as a silent failure.

    The refusal names the source, because the caller's next question is
    *which* one --- an ontology may have several and only one of them is the
    problem.
    """
    catalog = await _catalog(tmp_path)

    unenumerable = _with_source(catalog, _Unenumerable(None))
    with pytest.raises(ValidationError, match="products_table") as refused:
        EntitySourceIndexSource(unenumerable)
    assert "cannot enumerate" in str(refused.value)

    holds_none = EntitySourceIndexSource(_with_source(catalog, _Unenumerable(frozenset())))
    assert holds_none.declares() == frozenset({"catalog"})
    assert [item async for item in holds_none.stream_items()] == []


# --------------------------------------------------------------------------
# The undeclared type, and the regime that has no schema
# --------------------------------------------------------------------------


async def test_a_type_the_schema_does_not_declare_is_refused_only_where_a_schema_exists(
    tmp_path: Path,
) -> None:
    """**Both clauses**, and the negative one is what is being asserted.

    A document declaring no ``entity_types:`` at all is not a document
    declaring an empty schema --- it is the by-reference regime, where rows
    live in somebody else's table and nobody wrote a schema over them. That is
    the regime this adapter most exists to index, so a refusal that fired
    there would forbid it.

    The guard's condition is therefore read off ``entity_types`` rather than
    assumed: an empty schema section is no schema.
    """
    catalog = await _catalog(tmp_path)

    undeclared = _with_source(catalog, _Unenumerable(frozenset({"Product", "Phantom"})))
    with pytest.raises(ValidationError, match="Phantom"):
        EntitySourceIndexSource(undeclared)

    schemaless_path = tmp_path / "bare.yaml"
    schemaless_path.write_text(
        "ontology:\n  id: bare\n  version: '1.0'\n"
        "  entities:\n    - {id: row-1, type: Whatever, name: Row One}\n"
    )
    bare = await async_load_ontology(schemaless_path)
    assert bare.entity_types == {}

    source = EntitySourceIndexSource(bare)
    [item] = [item async for item in source.stream_items()]
    assert item.id == "bare:row-1"


# --------------------------------------------------------------------------
# The enumeration itself
# --------------------------------------------------------------------------


async def test_every_entity_is_yielded_once_with_a_qualified_id(tmp_path: Path) -> None:
    """The three entities of two types, each once, each qualified.

    The vocabulary declares a third type holding nothing, so this also asserts
    that a declared-but-empty type contributes no item and raises no error ---
    which is what tells *enumerate the schema* apart from *enumerate what is
    held*.
    """
    catalog = await _catalog(tmp_path)
    source = EntitySourceIndexSource(catalog)

    items = [item async for item in source.stream_items()]

    assert sorted(item.id for item in items) == [
        "catalog:acme",
        "catalog:sku-4471",
        "catalog:sku-8802",
    ]
    assert all(catalog.localize(item.id) for item in items)


async def test_the_metadata_is_the_ontology_id_and_the_forms_and_nothing_else(
    tmp_path: Path,
) -> None:
    """Two keys, both published constants, and no third.

    The forms travel in metadata rather than being expanded here: expanding
    them is a decorator's job, and a leaf source that did it would leave the
    decorator nothing to read.
    """
    catalog = await _catalog(tmp_path)
    source = EntitySourceIndexSource(catalog)

    by_id = {item.id: item async for item in source.stream_items()}

    widget = by_id["catalog:sku-4471"]
    assert set(widget.metadata) == {ONTOLOGY_ID_KEY, ALIAS_FORMS_KEY}
    assert widget.metadata[ONTOLOGY_ID_KEY] == "catalog"
    assert widget.metadata[ALIAS_FORMS_KEY] == ["Widget", "ACME widget"]
    assert by_id["catalog:sku-8802"].metadata[ALIAS_FORMS_KEY] == []


async def test_a_join_over_one_non_empty_value_leaves_no_dangling_separator(
    tmp_path: Path,
) -> None:
    """Measured rather than anticipated, over a vocabulary where it happens.

    Two of the three entities carry a name and no description, so the
    one-value case is the ordinary path here rather than an edge one --- and
    a separator applied after each value instead of between them would be
    invisible over a document where every row has both fields.
    """
    catalog = await _catalog(tmp_path)
    source = EntitySourceIndexSource(catalog, fields=("name", "description"))

    by_id = {item.id: item.text async for item in source.stream_items()}

    assert by_id["catalog:sku-4471"] == "Acme Widget -- a widget"
    assert by_id["catalog:sku-8802"] == "Bolt"
    assert by_id["catalog:acme"] == "Acme Corp"


async def test_a_field_an_entity_does_not_carry_is_refused_at_construction(
    tmp_path: Path,
) -> None:
    """Named where the caller is still holding the mistake.

    The alternative is a stream yielding empty text for every row, which
    reads downstream as an empty vocabulary rather than as a typo.
    """
    catalog = await _catalog(tmp_path)

    with pytest.raises(ValidationError, match="latin_name"):
        EntitySourceIndexSource(catalog, fields=("latin_name",))
    with pytest.raises(ValidationError, match="name, description"):
        EntitySourceIndexSource(catalog, fields=())


async def test_the_declared_set_is_the_ontology_id(tmp_path: Path) -> None:
    """One named set, which is what a scope filter is translated against."""
    catalog = await _catalog(tmp_path)
    assert EntitySourceIndexSource(catalog).declares() == frozenset({"catalog"})


async def test_the_stream_reads_the_backend_in_batches(tmp_path: Path) -> None:
    """``get_many`` batches its reads and returns everything; this batches the asks.

    The distinction is the whole reason the batching is the adapter's: handing
    a bulk read the entire union materialises the vocabulary inside a member
    called ``stream_items``, which is exactly the property the protocol exists
    to provide.
    """
    from dataknobs_common.ontology import index_source as module

    catalog = await _catalog(tmp_path)
    asked: list[int] = []

    class _Counting:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def describe(self) -> SourceDescription:
            return self._inner.describe()

        async def by_type(self, type_id: str) -> frozenset[str]:
            return await self._inner.by_type(type_id)

        async def get_many(self, entity_ids: Any) -> dict[str, Any]:
            asked.append(len(list(entity_ids)))
            return await self._inner.get_many(entity_ids)

    counting = _with_source(catalog, _Counting(catalog.entities))
    original = module.STREAM_BATCH_SIZE
    module.STREAM_BATCH_SIZE = 1
    try:
        items = [item async for item in EntitySourceIndexSource(counting).stream_items()]
    finally:
        module.STREAM_BATCH_SIZE = original

    assert len(items) == 3
    assert asked == [1, 1, 1], "the whole union reached the backend in one ask"


async def test_the_enumeration_streamed_is_the_one_validated_at_construction(
    tmp_path: Path,
) -> None:
    """The refusal at construction has to bind the read, or it refuses nothing.

    ``__post_init__`` refuses a source answering ``declares is None``, on the
    argument that an index over it *"would be silently partial"* and that an
    entity absent from an index resolves to nothing. ``stream_items`` then
    asked the same question a second time and wrote ``or frozenset()`` over
    the answer --- so a source that answered a set at construction and
    ``None`` at the read produced exactly the empty index the refusal exists
    to prevent, and the second refusal (a type the schema does not declare)
    was bypassed on the same path.

    Only a structural implementation can reach this, because both in-tree
    ``describe()`` implementations answer a non-``None`` set --- which is the
    population the three-state field was added for.
    """
    import dataclasses

    catalog = await _catalog(tmp_path)

    class _ForgetsWhatItHolds:
        """Enumerable when asked at construction, not when asked again."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self._asked = 0

        def describe(self) -> SourceDescription:
            self._asked += 1
            described = self._inner.describe()
            if self._asked == 1:
                return described
            return dataclasses.replace(described, declares=None)

        async def by_type(self, type_id: str) -> frozenset[str]:
            return await self._inner.by_type(type_id)

        async def get_many(self, entity_ids: Any) -> dict[str, Any]:
            return await self._inner.get_many(entity_ids)

    forgetful = _with_source(catalog, _ForgetsWhatItHolds(catalog.entities))
    source = EntitySourceIndexSource(forgetful)

    items = [item async for item in source.stream_items()]

    assert sorted(item.id for item in items) == [
        "catalog:acme",
        "catalog:sku-4471",
        "catalog:sku-8802",
    ]


# --------------------------------------------------------------------------
# An index full of empty rows, which is worse than an empty index
# --------------------------------------------------------------------------

#: A vocabulary whose entities carry a name and no description.
#:
#: The ordinary shape of a live binding read through a projection that fills
#: one column and not another --- and the shape ``fields: ["description"]``
#: turns into a stream of empty text. Every entity here is perfectly good;
#: the *request* is what holds nothing.
NAMES_ONLY = """\
ontology:
  id: parts
  version: "1.0"

  entity_types:
    - id: Part

  entities:
    - {id: p1, type: Part, name: Gear Pump}
    - {id: p2, type: Part, name: Gate Valve}
    - {id: p3, type: Part, name: Check Valve}
"""


async def test_a_build_whose_every_row_is_empty_text_is_reported_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A correctly spelled field the entities carry nothing under.

    :data:`TEXT_FIELDS`' construction check catches a *misspelled* name and
    says why --- *"the alternative is a stream that yields empty text for
    every row and looks like an empty vocabulary"*. It cannot catch a
    correctly spelled one, which is the ordinary case for a live binding: a
    projection fills whatever columns the consumer's table has.

    What that produces is not an empty index. It is an index full of rows
    equidistant from every query forever, which is worse, because an empty
    index has a report and this has an answer. So the stream reports it ---
    once per build, naming the fields and the source, at the level the
    neighbouring partial-coverage case already uses.
    """
    path = tmp_path / "parts.yaml"
    path.write_text(NAMES_ONLY)
    ontology = await async_load_ontology(path)
    source = EntitySourceIndexSource(ontology, fields=("description",))

    with caplog.at_level("WARNING"):
        items = [item async for item in source.stream_items()]

    assert len(items) == 3
    assert all(item.text == "" for item in items)
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1, "reported once per build, not once per row"
    message = warnings[0].getMessage()
    assert "description" in message
    assert "parts" in message


async def test_one_row_carrying_text_is_enough_to_report_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The report is about *every* row, and a partial fill is not this defect.

    The positive control for the test above: a vocabulary where one of three
    entities carries a description is a binding that holds the field on some
    rows, which is legitimate and common. Reporting there would fire on the
    ordinary case and say nothing true --- the same argument the
    partial-coverage warning one level up makes for excluding authored
    sources.
    """
    path = tmp_path / "some.yaml"
    path.write_text(
        NAMES_ONLY.replace(
            "{id: p2, type: Part, name: Gate Valve}",
            "{id: p2, type: Part, name: Gate Valve, description: a gate valve}",
        )
    )
    ontology = await async_load_ontology(path)
    source = EntitySourceIndexSource(ontology, fields=("description",))

    with caplog.at_level("WARNING"):
        items = [item async for item in source.stream_items()]

    assert [item.text for item in items] == ["", "a gate valve", ""]
    assert [record for record in caplog.records if record.levelname == "WARNING"] == []


async def test_a_vocabulary_holding_no_entities_at_all_is_not_reported_here(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Nothing streamed is a different condition, and it already has a report.

    *Every row is empty* needs a row. A source enumerating nothing yields
    nothing, and the state that produces --- an empty index --- is the one
    the construction refusals and the coverage warning are about. Firing here
    too would attach this message to a condition it does not describe.
    """
    ontology = await _catalog(tmp_path)
    source = EntitySourceIndexSource(
        _with_source(ontology, _Unenumerable(frozenset())), fields=("description",)
    )
    # The construction-time coverage warning is a different report about a
    # different condition, and it legitimately fires here: a schema naming
    # three types over a binding holding none. This test is about what the
    # *stream* says, so the construction's record is dropped first.
    caplog.clear()

    with caplog.at_level("WARNING"):
        items = [item async for item in source.stream_items()]

    assert items == []
    assert [record for record in caplog.records if record.levelname == "WARNING"] == []

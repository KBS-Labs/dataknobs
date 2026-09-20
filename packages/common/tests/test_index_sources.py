# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The protocol's three pure sources, asserted in the package that declares them.

``index.py`` ships a protocol and three reference implementations over it, and
every test of them lived in ``dataknobs-data`` --- reached through a vector
store, an embedder and a built index. So the one package that can exercise
them with nothing but its own dependencies was not doing it, and the behaviour
that is purely this package's (what a decorator reads out of metadata, what a
callable-backed source accepts, what the family hashes like) was asserted only
where a failure would arrive wearing a store's clothes.

These need no store, no embedder and no ``dataknobs-data`` import --- which is
the same property ``test_ontology_index_source.py`` states for the adapter one
level down.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

from dataknobs_common.index import (
    AliasSource,
    AsyncIndexSource,
    CallableSource,
    IndexItem,
    MappingSource,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

ALIASES = "dk_alias_forms"


def _one(**metadata: object) -> CallableSource:
    """One item carrying whatever metadata a test wants to put on it."""

    async def rows() -> AsyncIterator[IndexItem]:
        yield IndexItem(id="catalog:acme", text="Acme Corp", metadata=dict(metadata))

    return CallableSource(rows)


# --------------------------------------------------------------------------
# The list-valued rule, applied at the key's only published reader
# --------------------------------------------------------------------------


async def test_a_bare_string_alias_is_one_form_rather_than_its_characters() -> None:
    """The rule is published on the key; this is the only thing that reads it.

    ``ALIAS_FORMS_KEY`` is declared list-valued, and the constant's own
    docstring states the consequence for a reader: *"a reader takes a bare
    string as one node rather than as its characters."* ``AliasSource``
    iterated the value directly, so a metadata value of ``"ACME"`` --- which
    is what a hand-written ``CallableSource`` produces, and a callable-backed
    source is the documented escape hatch precisely because a consumer writes
    its metadata --- yielded four one-character items.

    All four carry the entity's id, and a store keyed on id upserts, so the
    surviving row's text was ``"E"`` and the entity could no longer be found
    by its own name.
    """
    decorated = AliasSource(_one(**{ALIASES: "ACME"}), ALIASES)

    items = [item async for item in decorated.stream_items()]

    assert [item.text for item in items] == ["Acme Corp", "ACME"]
    assert {item.id for item in items} == {"catalog:acme"}


async def test_a_mapping_valued_alias_field_is_not_iterated_into_its_keys() -> None:
    """The same shape one type along, and the same one-item answer.

    A ``dict`` is iterable over its keys, so the unguarded loop turned a
    mapping into one item per key --- silently, and under the entity's id.
    A mapping is not a malformed list to be salvaged the way a bare string is:
    there is no reading under which its keys are surface forms. So it is
    logged and dropped, and the entity keeps its canonical text.
    """
    decorated = AliasSource(_one(**{ALIASES: {"short": "ACME"}}), ALIASES)

    assert [item.text async for item in decorated.stream_items()] == ["Acme Corp"]


async def test_a_list_of_forms_still_yields_one_item_per_form() -> None:
    """The ordinary path, so the coercion above cannot have flattened it."""
    decorated = AliasSource(_one(**{ALIASES: ["ACME", "Acme Corporation"]}), ALIASES)

    assert [item.text async for item in decorated.stream_items()] == [
        "Acme Corp",
        "ACME",
        "Acme Corporation",
    ]


async def test_an_absent_or_empty_alias_field_yields_the_canonical_text_alone() -> None:
    """An entity nobody has another name for is not an error."""
    assert [item.text async for item in AliasSource(_one(), ALIASES).stream_items()] == [
        "Acme Corp"
    ]
    empty = AliasSource(_one(**{ALIASES: []}), ALIASES)
    assert [item.text async for item in empty.stream_items()] == ["Acme Corp"]


async def test_a_form_equal_to_the_canonical_text_is_not_yielded_twice() -> None:
    """A vocabulary listing the name among its own aliases is common."""
    decorated = AliasSource(_one(**{ALIASES: ["Acme Corp", "ACME"]}), ALIASES)

    assert [item.text async for item in decorated.stream_items()] == ["Acme Corp", "ACME"]


# --------------------------------------------------------------------------
# The escape hatch accepts the four shapes it documents
# --------------------------------------------------------------------------


async def test_a_callable_source_drives_all_four_shapes_it_accepts() -> None:
    """The docstring names four; a test that drove one would leave three claimed."""

    def sync_iterable() -> list[IndexItem]:
        return [IndexItem(id="a", text="alpha")]

    async def async_iterable() -> AsyncIterator[IndexItem]:
        yield IndexItem(id="a", text="alpha")

    async def awaitable_of_sync() -> list[IndexItem]:
        return [IndexItem(id="a", text="alpha")]

    async def awaitable_of_async() -> AsyncIterator[IndexItem]:
        return async_iterable()

    for fn in (sync_iterable, async_iterable, awaitable_of_sync, awaitable_of_async):
        source = CallableSource(fn)
        assert [item.text async for item in source.stream_items()] == ["alpha"], fn.__name__


async def test_a_mapping_source_yields_in_key_order() -> None:
    """Two builds over one mapping write rows in one order, which costs nothing here."""
    source = MappingSource({"dog": "Dog", "beagle": "Beagle"})

    assert [(item.id, item.text) async for item in source.stream_items()] == [
        ("beagle", "Beagle"),
        ("dog", "Dog"),
    ]


# --------------------------------------------------------------------------
# What the family declares, and what it hashes like
# --------------------------------------------------------------------------


def test_a_bare_source_declares_nothing_and_a_decorator_forwards_it() -> None:
    """``frozenset()`` is a complete answer: local ids fall in no named space.

    The decorator forwards rather than re-declaring, because a second
    translation site for a scope filter is a second thing to disagree.
    """
    assert MappingSource({"a": "alpha"}).declares() == frozenset()
    assert MappingSource({"a": "alpha"}, frozenset({"catalog"})).declares() == frozenset(
        {"catalog"}
    )
    inner = CallableSource(lambda: [], frozenset({"catalog"}))
    assert AliasSource(inner, ALIASES).declares() == frozenset({"catalog"})


def test_each_source_satisfies_the_protocol_it_is_a_reference_implementation_of() -> None:
    """A smoke test, which is what ``runtime_checkable`` offers and all it offers."""
    for source in (
        MappingSource({"a": "alpha"}),
        CallableSource(lambda: []),
        AliasSource(CallableSource(lambda: []), ALIASES),
    ):
        assert isinstance(source, AsyncIndexSource), type(source).__name__


def test_a_source_that_claims_hashable_can_actually_be_hashed() -> None:
    """``eq=False`` is the ruling; this is the property it was taken for.

    ``MappingSource`` holds a ``Mapping`` and ``AliasSource`` holds whatever
    its inner source is, so frozen with equality left on would generate a
    ``__hash__`` over the field tuple, answer ``Hashable`` and raise at the
    call --- the one combination that guards a caller against nothing.
    """
    for source in (
        MappingSource({"a": "alpha"}),
        CallableSource(lambda: []),
        AliasSource(CallableSource(lambda: []), ALIASES),
    ):
        assert isinstance(source, Hashable), type(source).__name__
        hash(source)

    # Identity rather than field-wise: two sources over one mapping are
    # interchangeable and nothing asks whether they are equal.
    assert MappingSource({"a": "alpha"}) != MappingSource({"a": "alpha"})


def test_an_index_item_compares_by_value_and_refuses_to_be_hashed() -> None:
    """The item is a record two of which really can be equal, so it keeps equality.

    Which makes ``__hash__`` ``None``, so the check answers False honestly
    rather than True and then raising --- the other of the two correct answers.
    """
    assert IndexItem(id="a", text="alpha") == IndexItem(id="a", text="alpha")
    assert not isinstance(IndexItem(id="a", text="alpha"), Hashable)

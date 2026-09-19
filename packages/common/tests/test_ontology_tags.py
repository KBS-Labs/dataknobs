# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The keys an indexed row is written under, the read of them, and what it refuses.

A value here is the one thing in this family that a corpus gets written under,
and after that it cannot change without migrating somebody else's rows. So the
values are asserted **literally** rather than read back from the constants: a
test that reads a constant to check that constant asserts nothing at all.

The read is the rest of the file. Every assertion below runs with **no store,
no embedder and no event loop** --- which is not a happy accident of how the
tests are written but the module's own rule, and the reason the whole contract
is checkable against nothing but a hand-edited file.

**What is deliberately not here.** The published call site is a workspace
guard, ``tests/test_worked_content_tags_call_site.py``, because a test living
inside this package can reach a name whether or not the door exports it and so
cannot fail for the one reason that test exists.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

from dataknobs_common.exceptions import NotFoundError, ValidationError
from dataknobs_common.ontology import (
    Assertion,
    EntityRef,
    MappingAssertionSource,
    MappingEntitySource,
    Ontology,
    TaxonomyDefinition,
    load_ontology,
    qualify,
)
from dataknobs_common.ontology import tags
from dataknobs_common.ontology.tags import (
    NODE_ID_KEY,
    ONTOLOGY_ID_KEY,
    TAXONOMY_ID_KEY,
    MalformedRow,
    NodeTag,
    TagReading,
    read_node_tags,
    read_node_tags_many,
)

from _vocabularies import MAMMALS_GUIDE_DOCUMENT, Sku, SkuCodec

#: The three keys of a well-formed tag, with the node list left to each caller.
TAGGED = {ONTOLOGY_ID_KEY: "mammals", TAXONOMY_ID_KEY: "species"}


def _row(node_id: object) -> dict[str, object]:
    """A well-formed tagged row carrying ``node_id`` under the node key."""
    return {**TAGGED, NODE_ID_KEY: node_id}


@pytest.fixture
def mammals(tmp_path: Path) -> Ontology:
    """The vocabulary the guides publish, loaded from a file.

    The document rather than a constructed ontology, because every question
    below is one a reader asks of a file they edited, and a vocabulary built in
    Python is a different object from the one a reader would have.
    """
    path = tmp_path / "mammals.yaml"
    path.write_text(MAMMALS_GUIDE_DOCUMENT, encoding="utf-8")
    return load_ontology(path)


@pytest.fixture
def lines() -> Ontology:
    """A vocabulary whose keys are **not** strings, with no entities at all.

    Two criteria need one: the round trip through ``localize``, and the claim
    that a tag's ``node_id`` stays a string where the keys are not. Over
    ``str`` both are untestable by construction --- ``StrCodec`` is the
    identity, so every parse succeeds and the two types are one type.

    **Its entity source is empty, and that is forced rather than chosen.** The
    concrete entity sources are declared over ``Mapping[str, Entity]`` and fold
    every id as text at construction, so a mapping keyed by a ``Sku`` raises
    before the vocabulary exists. Neither criterion needs an entity: one asks
    what ``at()`` *takes* and the other asks what type the two halves are, and
    the structure axis comes off the assertions rather than off the entities.
    """
    return Ontology(
        id="acme",
        version="1.0",
        entity_types={},
        relation_types={},
        entities=MappingEntitySource({}),
        assertions=MappingAssertionSource(
            [
                Assertion(
                    id="line-3-is-a-line-1",
                    subject=Sku("ACME", 3),
                    relation="isa",
                    object=EntityRef(Sku("ACME", 1)),
                )
            ]
        ),
        taxonomies={"lines": TaxonomyDefinition(id="lines", relation="isa")},
        describes=(),
        codec=SkuCodec(),  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# The keys
# --------------------------------------------------------------------------


def test_alias_forms_keys_value_is_asserted_literally() -> None:
    """``dk_alias_forms``, spelled out.

    Not ``startswith("dk_")`` and not ``ALIAS_FORMS_KEY == ALIAS_FORMS_KEY``.
    The prefix keeps the key out of a consumer's own namespace in a store we
    do not own; the underscore is what the other three already use, and a
    family spelled two ways is the defect a published family exists to
    prevent. The whole derivation is mechanical --- the constant's name less
    ``_KEY``, lower-cased --- and this is what makes that checkable.
    """
    assert tags.ALIAS_FORMS_KEY == "dk_alias_forms"


def test_the_other_three_values_are_asserted_literally_too() -> None:
    """The other three, on the same terms and for the same reason.

    They land in the same commit as ``ALIAS_FORMS_KEY`` and are free to change
    for exactly as long, so they are pinned here rather than left to whichever
    later work reads them back.
    """
    assert tags.ONTOLOGY_ID_KEY == "dk_ontology_id"
    assert tags.TAXONOMY_ID_KEY == "dk_taxonomy_id"
    assert tags.NODE_ID_KEY == "dk_node_id"


def test_the_module_publishes_every_name_it_defines_and_nothing_more() -> None:
    """``__all__`` is the module's surface, and the module has no other.

    **The membership list is the half that moved.** It used to be the four
    constants and the assertion said so, on the reading that a module holding
    the keys and not the read is one a leg needing only the keys can import
    cheaply. That reading was correct while it held and the read landing here
    is what the split scheduled, so the list grows rather than the guard going
    away.

    **The other half is what actually catches something and is unchanged**: a
    name implemented and not published. That is the failure a workspace guard
    over the door cannot see either, because a name missing from ``__all__``
    is missing from both.

    Imported names are excluded by where they were defined rather than by a
    list of exemptions: the module has imports now, and an exemption list would
    have to grow with them, which makes the guard weaker every time it is
    edited.
    """
    assert set(tags.__all__) == {
        "ALIAS_FORMS_KEY",
        "NODE_ID_KEY",
        "ONTOLOGY_ID_KEY",
        "TAXONOMY_ID_KEY",
        "MalformedRow",
        "NodeTag",
        "TagReading",
        "read_node_tags",
        "read_node_tags_many",
    }
    public = {
        name
        for name, value in vars(tags).items()
        if not name.startswith("_") and getattr(value, "__module__", tags.__name__) == tags.__name__
    }
    assert public == set(tags.__all__)


def test_the_two_halves_of_the_key_family_name_each_other() -> None:
    """The cross-reference the family split is paid for with.

    Four identity keys live here and a fifth --- the model that produced the
    vector --- lives in ``dataknobs_data.vector.content``, because that one is
    about an embedder, already ships, and has importers. The split is
    legitimate and its cost is a reader reaching for one module and finding
    one of five, so each docstring names the other. A cross-reference nothing
    checks is a cross-reference that silently goes away.

    Read off disk rather than through an import, because ``dataknobs-common``
    must not import ``dataknobs-data`` --- the whole point of the placement
    rule is that this package installs without it.

    **Which is also why the second half skips rather than fails when the
    sibling tree is absent.** Reaching it is a hard-coded relative offset into
    the workspace checkout, so running this package's suite against an
    installed wheel --- the very arrangement the placement rule exists to keep
    possible --- found no file and failed the assertion, asserting the
    opposite of what the docstring above claims. The forward half needs no
    sibling and is checked unconditionally.
    """
    here = Path(tags.__file__)
    assert "vector/content.py" in here.read_text()

    content = here.parents[4] / "data" / "src" / "dataknobs_data" / "vector" / "content.py"
    if not content.exists():
        pytest.skip(f"the sibling package tree is not on disk at {content}")
    assert "ontology/tags.py" in content.read_text()


# --------------------------------------------------------------------------
# The read: what it answers
# --------------------------------------------------------------------------


def test_a_tagged_row_reads_into_one_tag_per_node_with_no_store_or_loop() -> None:
    """The first line of the hop, and the one that is service-free by inspection.

    Two nodes in, two tags out, **in the order the row wrote them** --- a tag
    set is not a set, because a row's first node is usually the one it is most
    about. Nothing is constructed but the tags: no ontology is loaded, no
    backend is named, and the call is not a coroutine, so there is no loop for
    it to be on.
    """
    metadata = {
        ONTOLOGY_ID_KEY: "mammals",
        TAXONOMY_ID_KEY: "species",
        NODE_ID_KEY: ["golden_retriever", "beagle"],
        "model_name": "all-MiniLM-L6-v2",
    }

    read = read_node_tags(metadata)

    assert read == (
        NodeTag("mammals", "species", "golden_retriever"),
        NodeTag("mammals", "species", "beagle"),
    )
    assert not inspect.iscoroutinefunction(read_node_tags)


def test_a_row_with_none_of_the_keys_answers_empty_and_one_with_some_refuses() -> None:
    """Both clauses in one test, because the difference between them is the contract.

    An untagged row is the common case --- most of a corpus touched no
    vocabulary --- and answering it with an exception would make every caller
    wrap the common path in a ``try``, which catches the uncommon one too. A
    half-written row is a writer that got the contract wrong, and it is the
    only thing here that can be fixed by telling somebody.

    Asserted as one test because a suite that checked the empty answer
    somewhere and the refusal somewhere else would still pass with a read that
    answered ``()`` to both.
    """
    assert read_node_tags({"invoice_id": "2291"}) == ()
    assert read_node_tags({}) == ()

    with pytest.raises(ValidationError) as refusal:
        read_node_tags({NODE_ID_KEY: ["beagle"]})

    assert ONTOLOGY_ID_KEY in str(refusal.value)
    assert TAXONOMY_ID_KEY in str(refusal.value)


def test_a_bare_string_is_one_node_and_a_bytes_and_a_mapping_are_neither() -> None:
    """One test, three inputs, because they are three doors onto one failure.

    A bare ``str`` satisfies *iterate it* and must not be iterated: ``beagle``
    is one node, not six letters. So does a ``bytes``, which is a sequence of
    integers and would be six tags naming six numbers; and so, in a different
    way, does a ``Mapping``, which reads as its keys and would have
    **succeeded** on a shape nobody meant.

    The success and the two refusals are asserted together because the first
    is what makes the other two non-obvious: the rule that admits the bare
    string is exactly the rule that admits the other two.
    """
    assert read_node_tags(_row("beagle")) == (NodeTag("mammals", "species", "beagle"),)

    for written in (b"beagle", bytearray(b"beagle"), {"beagle": 1}, {"beagle"}):
        with pytest.raises(ValidationError) as refusal:
            read_node_tags(_row(written))
        assert NODE_ID_KEY in str(refusal.value)
        assert type(written).__name__ in str(refusal.value)


def test_a_scalar_that_is_not_a_string_raises_the_documented_class() -> None:
    """Asserted on the **class**, because what it replaces already raised.

    ``list(None)`` is a ``TypeError``, and a caller holding this function's
    documented ``Raises:`` catches nothing at all for the commonest
    serialization accident there is. So ``pytest.raises(ValidationError)`` and
    never ``Exception``: the point is not that something goes wrong, it is
    that what goes wrong is the class this contract names.
    """
    for written in (None, 42, 3.5, True):
        with pytest.raises(ValidationError) as refusal:
            read_node_tags(_row(written))
        assert NODE_ID_KEY in str(refusal.value)
        assert type(written).__name__ in str(refusal.value)

    with pytest.raises(ValidationError):
        read_node_tags({**_row(["beagle"]), ONTOLOGY_ID_KEY: 42})


def test_one_bad_entry_refuses_the_row_whole_and_three_are_all_named() -> None:
    """Both clauses in one test, one level down from the batch's own reasoning.

    A row declaring three nodes of which one is wrong must not count as
    evidence for two: that is a false statement about what the row is about,
    made silently, which is the failure this whole family exists to prevent.
    So the readable entries are not returned.

    And every offending position is named rather than the first, asserted over
    three rather than one, because a message naming the first and a message
    naming every one are indistinguishable at one.
    """
    with pytest.raises(ValidationError) as one:
        read_node_tags(_row(["golden_retriever", None, "beagle"]))

    assert "1 of 3" in str(one.value)

    with pytest.raises(ValidationError) as three:
        read_node_tags(_row([None, "beagle", 42, b""]))

    assert "3 of 4" in str(three.value)
    for position in ("0 is NoneType", "2 is int", "3 is bytes"):
        assert position in str(three.value)


def test_the_read_imports_nothing_outside_this_package_and_the_standard_library() -> None:
    """The dependency claim written as a test rather than as a paragraph.

    The keys and the read are in ``common`` because three strings and a pure
    function over a mapping have no third-party dependency --- which is what
    makes the whole of this file runnable with nothing installed but this
    package. An import of ``dataknobs_data`` arriving here by a later edit
    would make that false and nothing else would report it.

    Read out of the module's source rather than off ``sys.modules``: what is
    being asserted is what the file says, and a transitively imported module
    is somebody else's business.
    """
    reached: set[str] = set()
    for node in ast.walk(ast.parse(Path(tags.__file__).read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            reached.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            reached.update(alias.name.split(".")[0] for alias in node.names)

    assert reached
    outside = {
        name
        for name in reached
        if name != "dataknobs_common" and name not in sys.stdlib_module_names
    }
    assert not outside, f"{sorted(outside)} is neither this package nor the standard library"


# --------------------------------------------------------------------------
# The read: over a corpus
# --------------------------------------------------------------------------


def test_the_batch_answers_one_entry_per_position_and_names_the_malformed_one() -> None:
    """One entry per position, and the two empty entries that mean opposite things.

    Six rows: four readable, one that touched no vocabulary, one a writer got
    wrong. The reading holds six entries because the reply is positional ---
    dropping the refused one would shift every later row onto the wrong
    position, which is a defect that reports nothing.

    The untagged row and the malformed row **both** hold ``()``, and only the
    second appears in ``malformed``. That is asserted here rather than in a
    test of its own because either half alone reads as a shape rather than as
    a distinction: a reading with one field would have spelled *touched no
    vocabulary* and *was written wrong* the same way.
    """
    rows: list[dict[str, object]] = [
        _row(["golden_retriever"]),
        {"invoice_id": "2291"},
        _row(["beagle", "dog"]),
        _row("retriever"),
        _row(["mammal"]),
        {ONTOLOGY_ID_KEY: "mammals", NODE_ID_KEY: ["beagle"]},
    ]

    reading = read_node_tags_many(rows)

    assert len(reading.tags) == 6
    assert [len(row) for row in reading.tags] == [1, 0, 2, 1, 1, 0]
    assert reading.tags[1] == () and reading.tags[5] == ()
    assert reading.malformed == (MalformedRow(row=5, reason=reading.malformed[0].reason),)
    assert TAXONOMY_ID_KEY in reading.malformed[0].reason
    assert not inspect.iscoroutinefunction(read_node_tags_many)


def test_an_empty_node_list_is_untagged_rather_than_malformed() -> None:
    """The one value that must stay on the *untagged* side of the distinction.

    All three keys are present and every entry is well typed, vacuously. The
    key is list-valued because a row may be about several nodes, and a count
    has zero among its values --- so *this row is about no node of this axis*
    is a statement a writer can make, and ``()`` is what it means.

    Asserted through the batch reader beside the distinction it must not
    disturb, rather than against the row reader alone: what would be wrong is
    not the empty answer but a ``MalformedRow`` appearing beside it.
    """
    reading = read_node_tags_many([_row([]), _row(["beagle"])])

    assert reading.tags == ((), (NodeTag("mammals", "species", "beagle"),))
    assert reading.malformed == ()
    assert read_node_tags(_row([])) == ()


def test_require_readable_returns_the_tags_over_a_clean_reading_and_refuses_over_any() -> None:
    """Both clauses in one test, on the empty-answer precedent above.

    The difference between them is the contract. A reading with nothing wrong
    hands back exactly what it holds --- not a copy, not a filtered view ---
    and a reading with anything wrong refuses. A suite asserting only the
    second would pass against a member that always raised.
    """
    clean = read_node_tags_many([_row(["beagle"]), {"invoice_id": "2291"}])

    assert clean.require_readable() == clean.tags

    dirty = read_node_tags_many([_row(["beagle"]), {NODE_ID_KEY: ["dog"]}])

    with pytest.raises(ValidationError) as refusal:
        dirty.require_readable()

    assert "row 1" in str(refusal.value)
    assert TagReading(tags=()).require_readable() == ()


def test_a_refusal_over_three_malformed_rows_names_three_positions() -> None:
    """Three rather than one, because at one every message looks complete.

    A message naming the first bad row and a message naming every bad row are
    indistinguishable over a single failure --- and the difference between
    them is the whole of what the batch form buys over the row reader called
    in a loop.
    """
    rows: list[dict[str, object]] = [_row(["beagle"]) for _ in range(6)]
    rows[1] = {NODE_ID_KEY: ["dog"]}
    rows[3] = _row(b"beagle")
    rows[5] = _row(["dog", 42])

    reading = read_node_tags_many(rows)

    assert [bad.row for bad in reading.malformed] == [1, 3, 5]

    with pytest.raises(ValidationError) as refusal:
        reading.require_readable()

    message = str(refusal.value)
    assert "3 of 6" in message
    for position in ("row 1", "row 3", "row 5"):
        assert position in message


# --------------------------------------------------------------------------
# The tie-back-in: what only a vocabulary can judge
# --------------------------------------------------------------------------


def test_a_tags_id_round_trips_into_a_key_and_a_foreign_one_is_refused(
    lines: Ontology,
) -> None:
    """The guard and the pass-through are one behaviour, over a non-``str`` key.

    ``qualified_id`` is the member that says the two spellings of a namespaced
    id are the same thing: the pair the row carries, and the one string the
    vocabulary's door takes. What comes back out of ``localize`` is what
    ``at()`` takes --- which over ``str`` proves nothing, because the codec is
    the identity and every string parses. Over a key of a consumer's own it is
    a parse that can fail, so the round trip is a claim rather than a tautology.

    The foreign refusal is asserted in the same test as the success because
    only an ontology knows whose ids it is holding: a door that accepted
    anything would pass the first half of this test and lose the second.
    """
    tag = read_node_tags(
        {ONTOLOGY_ID_KEY: "acme", TAXONOMY_ID_KEY: "lines", NODE_ID_KEY: "ACME/3"}
    )[0]

    assert tag.qualified_id == qualify("acme", "ACME/3") == "acme:ACME/3"

    here = lines.taxonomy(tag.taxonomy_id).at(lines.localize(tag.qualified_id))

    assert here.node == Sku("ACME", 3)
    assert here.exists() is True
    assert [above.node for above in here.parents()] == [Sku("ACME", 1)]

    with pytest.raises(ValidationError) as refusal:
        lines.localize(qualify("procedures", "spay"))

    assert "procedures" in str(refusal.value) and "acme" in str(refusal.value)


def test_a_tags_node_id_is_a_string_even_where_the_axis_keys_are_not(
    lines: Ontology,
) -> None:
    """The two types in one expression, which is what makes the bar checkable.

    The read answers **rendered** ids --- keys in the space ``subtree_keys``
    returns --- and ``localize`` is what turns one into a key. So a tag's
    ``node_id`` is a ``str`` over a vocabulary whose keys are ``Sku``, and the
    thing the cursor takes is not.

    This is the falsifier for the string bar the read enforces: if a tag's
    entries were ever keys rather than rendered ids, the bar would be wrong and
    would have to be *whatever the codec parses*. It holds for exactly as long
    as ``node_id`` is documented as rendered.
    """
    tag = read_node_tags(
        {ONTOLOGY_ID_KEY: "acme", TAXONOMY_ID_KEY: "lines", NODE_ID_KEY: "ACME/3"}
    )[0]
    key = lines.localize(tag.qualified_id)

    assert isinstance(tag.node_id, str)
    assert not isinstance(key, str)
    assert key == Sku("ACME", 3)
    assert lines.taxonomy("lines").subtree_keys(Sku("ACME", 1)) == [
        Sku("ACME", 1),
        Sku("ACME", 3),
    ]


def test_a_node_not_carried_an_axis_not_declared_and_the_accessor_that_still_refuses(
    mammals: Ontology,
) -> None:
    """One operational fact one level apart, and the refusal that is not it.

    A tag naming a node this vocabulary does not carry answers ``exists()``
    ``False`` and raises nothing: the read judged nothing about a vocabulary it
    was never given, and *nothing below this node* stays a different answer
    from *this node is not here*.

    A tag naming an **axis** this vocabulary does not declare is the same fact
    one level up, and the caller filters it --- a name off a row is not a name
    anybody typed, so there is nobody to tell. The filter builds nothing: it is
    a membership test on a mapping the vocabulary already holds.

    And the accessor still refuses, asserted beside the silence. Without that
    half this test would pass against a vocabulary that had stopped refusing
    anything at all, and the filter would look like a behaviour that went away
    rather than a choice the caller made.
    """
    axis = mammals.taxonomy("species")
    stale = read_node_tags(_row(["no_such_node"]))[0]

    assert axis.at(stale.node_id).exists() is False

    drifted = read_node_tags(
        {ONTOLOGY_ID_KEY: "mammals", TAXONOMY_ID_KEY: "coat", NODE_ID_KEY: ["double_coat"]}
    )
    mine = [
        tag
        for tag in drifted
        if tag.ontology_id == mammals.id and tag.taxonomy_id in mammals.taxonomies
    ]

    assert drifted and mine == []

    with pytest.raises(NotFoundError) as refusal:
        mammals.taxonomy("coat")

    assert "species" in str(refusal.value)


# --------------------------------------------------------------------------
# What the module documents about itself
# --------------------------------------------------------------------------


def test_every_member_that_raises_names_what_escapes_it() -> None:
    """A refusal is documented where it escapes, read off ``__doc__``.

    Asserted over the members **by name** rather than over the module, because
    which of them raise is a property of the design rather than something a
    test should discover --- and asserted **negatively** too, on the two that
    raise nothing and must keep saying nothing. A test that only checked for
    the presence of a section would pass a module that had grown one
    everywhere.
    """
    for member in (read_node_tags, TagReading.require_readable):
        doc = inspect.getdoc(member) or ""
        assert "Raises:" in doc, f"{member.__qualname__} documents no refusal"
        assert "ValidationError" in doc

    for quiet in (read_node_tags_many, NodeTag.qualified_id.fget):
        doc = inspect.getdoc(quiet) or ""
        assert "Raises:" not in doc, (
            "this member raises nothing, so a Raises: section here is either "
            "false or a refusal that arrived without this guard noticing"
        )

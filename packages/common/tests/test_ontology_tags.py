# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The keys an indexed row is written under, and the module split that holds them.

A value here is the one thing in this family that a corpus gets written under,
and after that it cannot change without migrating somebody else's rows. So the
values are asserted **literally** rather than read back from the constants: a
test that reads a constant to check that constant asserts nothing at all.
"""

from __future__ import annotations

from pathlib import Path

from dataknobs_common.ontology import tags


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


def test_the_module_carries_the_four_constants_and_nothing_else() -> None:
    """The read is not here, and the module says so by holding nothing else.

    A module that quietly grows the reader alongside the keys is one a leg
    needing only the keys has to import anyway.
    """
    assert set(tags.__all__) == {
        "ALIAS_FORMS_KEY",
        "NODE_ID_KEY",
        "ONTOLOGY_ID_KEY",
        "TAXONOMY_ID_KEY",
    }
    public = {
        name for name in vars(tags) if not name.startswith("_") and name not in {"annotations"}
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
    """
    here = Path(tags.__file__)
    assert "vector/content.py" in here.read_text()

    content = here.parents[4] / "data" / "src" / "dataknobs_data" / "vector" / "content.py"
    assert content.exists(), f"expected the other half of the family at {content}"
    assert "ontology/tags.py" in content.read_text()

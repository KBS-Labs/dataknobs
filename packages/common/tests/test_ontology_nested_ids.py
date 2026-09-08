"""A keyless tree gets ids; a file that declares its own keeps them.

The rule is *declared wins wherever it exists*, and the reason is not
tidiness. Minting into a file that already names its entities gives one node
two ids, and every authored assertion points at one of the two.
"""

from __future__ import annotations

import pytest

from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import load_ontology

PRODUCT_AREAS = {
    "id": "areas",
    "sources": [
        {
            "id": "product_areas",
            "kind": "nested",
            "tree": {
                "name": "Billing",
                "children": [
                    {
                        "name": "Invoices",
                        "children": [{"name": "Late Fees"}],
                    },
                    {"name": "Refunds"},
                ],
            },
        }
    ],
}


def test_a_keyless_tree_mints_an_id_per_node() -> None:
    """The id is a slug of the path, so it survives a rename of the node."""
    onto = load_ontology(PRODUCT_AREAS)

    assert onto.entities.by_type("product_areas") == frozenset(
        {
            "billing",
            "billing/invoices",
            "billing/invoices/late-fees",
            "billing/refunds",
        }
    )


def test_the_path_becomes_the_name() -> None:
    """The name is what changes when someone edits the file. The id is not."""
    onto = load_ontology(PRODUCT_AREAS)

    node = onto.entity("billing/invoices/late-fees")
    assert node is not None
    assert node.name == "Billing/Invoices/Late Fees"


def test_the_tree_edges_are_asserted() -> None:
    """A minted tree that carried no edges would be a list, not a hierarchy."""
    onto = load_ontology(PRODUCT_AREAS)

    edges = onto.assertions.find(subject="billing/invoices/late-fees")
    assert len(edges) == 1
    assert edges[0].object.entity_id == "billing/invoices"


def test_the_minted_assertion_ids_are_pinned() -> None:
    """The exact ids one document mints, asserted rather than implied.

    Everything about minting -- the traversal, the slug, the collision refusal
    -- is shared with ``MappingHierarchy.from_nested`` so that one tree read
    through either door yields one vocabulary. Sharing it moved the code, and a
    slug rule that shifts by one character silently re-keys every entity already
    minted and every assertion made against them. The failure is invisible in
    any test that only asks whether the document loaded, so the ids and the
    assertion ids are written out here.
    """
    onto = load_ontology(PRODUCT_AREAS)

    assert {a.id for a in onto.assertions.find(relation="isa")} == {
        "billing/invoices-isa-billing",
        "billing/invoices/late-fees-isa-billing/invoices",
        "billing/refunds-isa-billing",
    }


def test_a_root_gets_no_parent_edge() -> None:
    onto = load_ontology(PRODUCT_AREAS)

    assert onto.assertions.find(subject="billing") == []


def test_a_declaring_file_mints_nothing() -> None:
    """``entities:`` present means the tree references those ids."""
    onto = load_ontology(
        {
            "id": "areas",
            "entities": [
                {"id": "billing", "type": "Area", "name": "Billing"},
                {"id": "invoices", "type": "Area", "name": "Invoices"},
            ],
            "sources": [
                {
                    "id": "product_areas",
                    "kind": "nested",
                    "tree": {"id": "billing", "children": [{"id": "invoices"}]},
                }
            ],
        }
    )

    assert onto.entities.by_type("Area") == frozenset({"billing", "invoices"})
    assert onto.entity("billing").name == "Billing"


def test_a_tree_node_naming_an_undeclared_id_is_refused() -> None:
    """In the declared regime the tree is a reference, so a miss is a typo."""
    with pytest.raises(ValidationError) as excinfo:
        load_ontology(
            {
                "id": "areas",
                "entities": [{"id": "billing", "type": "Area"}],
                "sources": [
                    {
                        "id": "product_areas",
                        "kind": "nested",
                        "tree": {"id": "billing", "children": [{"id": "invocies"}]},
                    }
                ],
            }
        )

    assert "'invocies'" in str(excinfo.value)


def test_custom_child_and_name_keys_are_honoured() -> None:
    """The defaults are defaults, not the only spelling a file may use."""
    onto = load_ontology(
        {
            "id": "areas",
            "sources": [
                {
                    "id": "areas",
                    "kind": "nested",
                    "child_key": "sub",
                    "name_key": "title",
                    "tree": {"title": "Top", "sub": [{"title": "Under"}]},
                }
            ],
        }
    )

    assert onto.entity("top/under") is not None


def test_a_list_of_roots_is_a_forest() -> None:
    """A tree with several roots is an ordinary shape, not a malformed one."""
    onto = load_ontology(
        {
            "id": "areas",
            "sources": [
                {
                    "id": "areas",
                    "kind": "nested",
                    "tree": [{"name": "One"}, {"name": "Two"}],
                }
            ],
        }
    )

    assert onto.entities.by_type("areas") == frozenset({"one", "two"})


def test_the_edge_relation_is_configurable() -> None:
    onto = load_ontology(
        {
            "id": "areas",
            "sources": [
                {
                    "id": "areas",
                    "kind": "nested",
                    "relation": "part_of",
                    "tree": {"name": "Top", "children": [{"name": "Under"}]},
                }
            ],
        }
    )

    assert onto.assertions.find(subject="top/under", relation="part_of")
    assert onto.assertions.find(subject="top/under", relation="isa") == []

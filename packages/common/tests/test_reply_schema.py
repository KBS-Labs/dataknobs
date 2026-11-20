"""Closed JSON Schemas for a model's reply.

Each builder's output is validated with ``jsonschema``: it is a well-formed
schema, it admits the reply it describes, and it refuses a reply that leaves a
property out, adds one, or gives one the wrong type. The fixture is an
accounts-payable reply that reads an invoice (``INV.2026.004``) into its notes
and lines. It is written for these tests: the invoice, SKUs and amounts are
invented and carry no real data.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Any, cast

import jsonschema
import pytest
from jsonschema import Draft202012Validator

import dataknobs_common
from dataknobs_common.exceptions import SerializationError
from dataknobs_common.fields import FieldType
from dataknobs_common.reply_schema import STRING, array, strict

#: An invoice line: the SKU billed and its amount, written as text.
LINE = strict({"sku": STRING, "amount": STRING})

#: An invoice read into its number, its vendor's notes and its lines.
INVOICE = strict({"number": STRING, "notes": array(STRING), "lines": array(LINE)})

#: A reply as a model would write it under :data:`INVOICE`.
READ = {
    "number": "INV.2026.004",
    "notes": ["net 30", "freight included"],
    "lines": [
        {"sku": "SKU-1042", "amount": "34.50"},
        {"sku": "SKU-2210", "amount": "12.00"},
    ],
}


def admits(schema: dict[str, Any], reply: Any) -> bool:
    """Whether ``schema``, checked as a schema first, admits ``reply``."""
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema).is_valid(reply)


def objects(schema: Any) -> list[dict[str, Any]]:
    """Every object schema inside ``schema``, ``schema`` first if it is one."""
    found: list[dict[str, Any]] = []
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            found.append(schema)
        for value in schema.values():
            found += objects(value)
    elif isinstance(schema, list):
        for value in schema:
            found += objects(value)
    return found


def containers(schema: Any) -> list[Any]:
    """Every dict and list inside ``schema``, ``schema`` included."""
    found: list[Any] = []
    if isinstance(schema, (dict, list)):
        found.append(schema)
        for value in schema.values() if isinstance(schema, dict) else schema:
            found += containers(value)
    return found


def test_a_string() -> None:
    assert STRING == {"type": "string"}
    assert admits(dict(STRING), "SKU-1042")
    assert not admits(dict(STRING), 1042)
    assert not admits(dict(STRING), None)


def test_an_array_holds_only_its_items() -> None:
    notes = array(STRING)

    assert notes == {"type": "array", "items": {"type": "string"}}
    assert admits(notes, ["net 30", "freight included"])
    assert admits(notes, [])
    assert not admits(notes, ["net 30", 30])
    assert not admits(notes, "net 30"), "a string is not an array of strings"
    assert not admits(notes, {"0": "net 30"})


def test_a_strict_object_requires_every_property_and_allows_no_other() -> None:
    assert LINE == {
        "type": "object",
        "properties": {"sku": {"type": "string"}, "amount": {"type": "string"}},
        "required": ["sku", "amount"],
        "additionalProperties": False,
    }
    assert admits(LINE, {"sku": "SKU-1042", "amount": "34.50"})
    assert admits(LINE, {"amount": "34.50", "sku": "SKU-1042"}), "the reply's order is free"
    assert not admits(LINE, {"sku": "SKU-1042"}), "a property left out"
    assert not admits(LINE, {"sku": "SKU-1042", "amount": "34.50", "qty": "2"}), "one added"
    assert not admits(LINE, {"sku": "SKU-1042", "amount": 34.5}), "an amount as a number"
    assert not admits(LINE, [["sku", "SKU-1042"], ["amount", "34.50"]]), "not an object"


def test_required_follows_the_order_the_properties_were_given() -> None:
    built = strict({"amount": STRING, "sku": STRING})
    assert built["required"] == ["amount", "sku"]
    assert list(built["properties"]) == ["amount", "sku"]


def test_any_property_name_can_be_written() -> None:
    """Names a keyword argument could not carry, and names that are keywords
    of JSON Schema itself, are properties like any other.
    """
    names = ["class", "unit-price", "2nd line", "type", "required", "properties"]
    built = strict(dict.fromkeys(names, STRING))

    assert built["required"] == names
    assert built["type"] == "object"
    reply = dict.fromkeys(names, "x")
    assert admits(built, reply)
    assert not admits(built, {**reply, "type": 1})
    assert not admits(built, {k: v for k, v in reply.items() if k != "required"})


def test_a_strict_object_inside_an_array_is_closed_too() -> None:
    assert admits(INVOICE, READ)
    for line in (
        {"sku": "SKU-2210"},
        {"sku": "SKU-2210", "amount": "12.00", "discount": "0.50"},
        {"sku": "SKU-2210", "amount": 12},
    ):
        reply = {**READ, "lines": [READ["lines"][0], line]}
        assert not admits(INVOICE, reply), line
    assert not admits(INVOICE, {k: v for k, v in READ.items() if k != "notes"})
    assert not admits(INVOICE, {**READ, "currency": "USD"})


@pytest.mark.parametrize(
    "schema",
    [
        strict({}),
        LINE,
        INVOICE,
        array(INVOICE),
        strict(
            {
                "batches": array(array(LINE)),
                "approved_by": array(strict({"name": STRING})),
            }
        ),
    ],
    ids=["empty", "line", "invoice", "invoices", "nested"],
)
def test_every_object_built_is_closed(schema: dict[str, Any]) -> None:
    """The form OpenAI's strict mode and Anthropic accept, at every depth."""
    Draft202012Validator.check_schema(schema)
    found = objects(schema)
    assert found
    for each in found:
        assert each["additionalProperties"] is False
        assert each["required"] == list(each["properties"])


def test_an_object_with_no_properties_admits_only_an_empty_one() -> None:
    assert admits(strict({}), {})
    assert not admits(strict({}), {"sku": "SKU-1042"})


def test_a_built_schema_is_json_and_is_not_changed_by_its_parts_afterwards() -> None:
    """A caller bounds a schema by building a new one, never by editing a part
    it passed in; and editing that part afterwards leaves what was built alone.
    """
    notes: dict[str, Any] = {"type": "string"}
    line: dict[str, Any] = {"type": "string"}
    built = strict({"notes": array(notes), "sku": line})
    notes["maxLength"] = 3
    line["enum"] = ["SKU-1042"]

    assert built == strict({"notes": array(STRING), "sku": STRING})
    assert json.loads(json.dumps(built)) == built
    jsonschema.validate({"notes": ["net 30"], "sku": "SKU-2210"}, built)


def test_a_part_edited_below_its_top_level_does_not_reach_the_schema() -> None:
    """The copy is deep: a one-level copy would share ``line['properties']``."""
    line = strict({"sku": STRING, "amount": STRING})
    invoice = strict({"number": STRING, "lines": array(line)})
    invoices = array(line)

    line["properties"]["amount"]["pattern"] = r"^\d+$"
    line["required"].append("qty")

    assert invoice == strict({"number": STRING, "lines": array(LINE)})
    assert invoices == array(LINE)
    assert admits(invoice, {"number": READ["number"], "lines": READ["lines"]})


def test_a_built_schema_shares_no_container_with_its_parts_or_another_schema() -> None:
    """One part reused in three schemas, as a layout composes them, is three
    objects; editing one schema in place reaches neither the part nor the others.
    """
    line = strict({"sku": STRING, "amount": STRING})
    built = [
        strict({"lines": array(line)}),
        array(line),
        strict({"first": line, "last": line}),
    ]

    source_ids = {id(each) for each in containers(line)}
    seen: set[int] = set()
    for schema in built:
        ids = [id(each) for each in containers(schema)]
        assert len(ids) == len(set(ids)), "a container shared inside one schema"
        assert not source_ids & set(ids), "a container shared with the part"
        assert not seen & set(ids), "a container shared with another schema"
        seen |= set(ids)

    built[2]["properties"]["first"]["properties"]["sku"]["maxLength"] = 8
    assert built[2]["properties"]["last"] == LINE
    assert built[1]["items"] == LINE
    assert line == LINE


def test_a_container_repeated_inside_one_hand_written_part_is_copied_twice() -> None:
    """Sharing in the caller's own part does not survive into the schema."""
    code: dict[str, Any] = {"type": "string", "pattern": "^SKU-"}
    built = array({"anyOf": [code, code]})

    first, second = built["items"]["anyOf"]
    assert first == second == code
    assert first is not second
    first["pattern"] = "^INV-"
    assert second["pattern"] == "^SKU-"


def test_string_cannot_be_changed() -> None:
    """Every schema built after such an edit would carry it."""
    with pytest.raises(TypeError):
        cast("Any", STRING)["format"] = "date"
    assert strict({"sku": STRING})["properties"]["sku"] == {"type": "string"}


def test_any_mapping_sequence_or_set_part_is_written_as_dict_and_list() -> None:
    """A part handed in frozen, as a tuple or as a set comes out as ``json``
    writes it: a set's members sorted, so the schema is the same on every run.
    """
    currency = MappingProxyType({"type": "string", "enum": ("USD", "EUR")})
    terms = {"type": "string", "enum": {"net 60", "net 30"}}
    built = strict({"currency": currency, "terms": terms, "lines": array(MappingProxyType(LINE))})

    assert type(built["properties"]["currency"]) is dict
    assert built["properties"]["currency"]["enum"] == ["USD", "EUR"]
    assert built["properties"]["terms"]["enum"] == ["net 30", "net 60"]
    assert type(built["properties"]["lines"]["items"]) is dict
    assert json.loads(json.dumps(built)) == built
    reply = {"currency": "USD", "terms": "net 30", "lines": READ["lines"]}
    assert admits(built, reply)
    assert not admits(built, {**reply, "currency": "GBP"})


@pytest.mark.parametrize(
    ("build", "named"),
    [
        (lambda: strict({"sku": "string"}), "property 'sku'"),
        (lambda: strict({"sku": FieldType.STRING}), "property 'sku'"),
        (lambda: array(cast("Any", str)), "array items"),
        (lambda: array(cast("Any", None)), "array items"),
        (lambda: strict(cast("Any", [("sku", STRING)])), "mapping of name to schema"),
        (lambda: strict(cast("Any", {1: STRING})), "names must be strings"),
    ],
    ids=["a-type-name", "a-field-type", "a-class", "none", "pairs", "a-non-string-name"],
)
def test_what_is_not_a_schema_mapping_is_refused(build: Any, named: str) -> None:
    """A slip that would otherwise reach the vendor as an invalid schema."""
    with pytest.raises(TypeError, match=named):
        build()


@pytest.mark.parametrize(
    ("part", "path"),
    [
        ({"type": "string", "default": object()}, "default"),
        ({"type": "number", "maximum": float("nan")}, "maximum"),
    ],
    ids=["an-object", "a-nan"],
)
def test_a_part_json_cannot_write_is_refused_naming_where(part: dict[str, Any], path: str) -> None:
    with pytest.raises(SerializationError, match="property 'amount'") as raised:
        strict({"sku": STRING, "amount": array(STRING) | {"items": part}})
    assert path in str(raised.value)
    assert raised.value.context["part"] == "property 'amount'"


def test_a_part_that_holds_itself_is_refused() -> None:
    part: dict[str, Any] = {"type": "object"}
    part["properties"] = {"next": part}
    with pytest.raises(SerializationError, match="array items"):
        array(part)


def test_the_builders_are_imported_from_their_module() -> None:
    """Not from the package door, where ``STRING`` would sit beside
    ``FieldType.STRING`` and ``array`` shadow the stdlib module under ``*``.
    """
    for name in ("STRING", "array", "strict"):
        assert name not in dataknobs_common.__all__

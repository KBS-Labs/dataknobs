"""Values as data, on a product catalogue's API replies.

A catalogue service answers with frozen dataclasses. A dashboard, a test or
another process wants the same reply as JSON, and wants to be told, not
shown an empty field, when part of it has no JSON form. The fixtures are
written for these tests: the products, suppliers and order are invented and
carry no real data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Any

import pytest

from dataknobs_common.exceptions import SerializationError
from dataknobs_common.serialization import as_data, fields_data


class Status(Enum):
    ACTIVE = "active"
    DISCONTINUED = "discontinued"


@dataclass(frozen=True)
class Supplier:
    supplier_id: str
    name: str

    def ref(self) -> str:
        return f"supplier:{self.supplier_id}"


@dataclass(frozen=True)
class PreferredSupplier(Supplier):
    """A subclass, for the ``using`` lookup order."""


@dataclass(frozen=True)
class Price:
    """A value that chose its own shape: an amount as text, never a float."""

    cents: int
    currency: str

    def to_dict(self) -> dict[str, Any]:
        whole, part = divmod(self.cents, 100)
        return {"amount": f"{whole}.{str(part).zfill(2)}", "currency": self.currency}


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    tags: frozenset[str]
    status: Status
    price: Price
    suppliers: tuple[Supplier, ...]
    attributes: MappingProxyType[str, str]
    stock: int
    weight_kg: float
    in_stock: bool
    replaced_by: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Order:
    order_id: str
    lines: tuple[Product, ...]
    placed_by: str


HARBOR = Supplier("S-07", "Harbor Metalworks")
KETTLE = Product(
    sku="SKU-1042",
    name="Stovetop kettle, 2 l",
    tags=frozenset({"stainless", "kitchen", "induction"}),
    status=Status.ACTIVE,
    price=Price(3450, "EUR"),
    suppliers=(HARBOR,),
    attributes=MappingProxyType({"capacity": "2 l", "finish": "brushed"}),
    stock=12,
    weight_kg=1.25,
    in_stock=True,
)


def test_a_reply_becomes_data_that_json_writes() -> None:
    data = as_data(KETTLE)
    assert data == {
        "sku": "SKU-1042",
        "name": "Stovetop kettle, 2 l",
        "tags": ["induction", "kitchen", "stainless"],
        "status": "active",
        "price": {"amount": "34.50", "currency": "EUR"},
        "suppliers": [{"supplier_id": "S-07", "name": "Harbor Metalworks"}],
        "attributes": {"capacity": "2 l", "finish": "brushed"},
        "stock": 12,
        "weight_kg": 1.25,
        "in_stock": True,
        "replaced_by": None,
        "notes": {},
    }
    assert list(data) == list(Product.__dataclass_fields__)
    assert json.loads(json.dumps(data)) == data


def test_a_set_is_a_sorted_list_whatever_order_it_was_built_in() -> None:
    """The dashboard case: a ``frozenset`` of tags is carried, not dropped."""
    one = as_data(KETTLE)
    other = as_data(
        Product(**{**KETTLE.__dict__, "tags": frozenset(["kitchen", "induction", "stainless"])})
    )
    assert one["tags"] == other["tags"] == ["induction", "kitchen", "stainless"]
    suppliers = {Supplier("S-12", "Linden Ceramics"), HARBOR}
    assert as_data(suppliers) == [
        {"supplier_id": "S-07", "name": "Harbor Metalworks"},
        {"supplier_id": "S-12", "name": "Linden Ceramics"},
    ]


def test_a_value_that_chose_its_own_shape_keeps_it() -> None:
    """``to_dict()`` is read before a dataclass's fields."""
    assert as_data(Price(905, "EUR")) == {"amount": "9.05", "currency": "EUR"}
    with pytest.raises(SerializationError, match="value is of type type"):
        as_data(Price)


def test_a_to_dict_that_raises_is_refused_where_it_is() -> None:
    class Unpriced:
        def to_dict(self) -> dict[str, Any]:
            raise ValueError("no price list loaded")

    with pytest.raises(
        SerializationError, match=r"value\['price'\]: to_dict\(\) of Unpriced raised ValueError"
    ) as caught:
        as_data({"price": Unpriced()})
    assert isinstance(caught.value.__cause__, ValueError)
    assert caught.value.context["path"] == "value['price']"


def test_a_type_can_be_shown_another_way_in_one_place() -> None:
    assert as_data(KETTLE, using={Supplier: Supplier.ref})["suppliers"] == ["supplier:S-07"]
    assert as_data(KETTLE)["suppliers"] == [{"supplier_id": "S-07", "name": "Harbor Metalworks"}]


def test_the_exact_type_is_used_before_a_base_listed_first() -> None:
    using = {Supplier: Supplier.ref, PreferredSupplier: lambda s: f"preferred:{s.supplier_id}"}
    assert as_data(PreferredSupplier("S-07", "Harbor"), using=using) == "preferred:S-07"
    assert as_data(HARBOR, using=using) == "supplier:S-07"


def test_a_conversion_is_applied_once_not_to_its_own_result() -> None:
    def renamed(supplier: Supplier) -> Supplier:
        return Supplier(supplier.supplier_id, supplier.name.upper())

    assert as_data(HARBOR, using={Supplier: renamed}) == {
        "supplier_id": "S-07",
        "name": "HARBOR METALWORKS",
    }


def test_some_fields_can_be_replaced() -> None:
    """A field too large to carry whole is carried as a count."""
    order = Order("ORD-88", (KETTLE,) * 200, "warehouse-2")
    data = fields_data(order, lines={"count": len(order.lines)})
    assert data == {"order_id": "ORD-88", "lines": {"count": 200}, "placed_by": "warehouse-2"}
    replaced = fields_data(order, lines=(KETTLE,))
    assert replaced["lines"][0]["tags"] == ["induction", "kitchen", "stainless"]


def test_a_replacement_must_name_a_field() -> None:
    order = Order("ORD-88", (), "warehouse-2")
    with pytest.raises(SerializationError, match="Order has no field items"):
        fields_data(order, items=())


def test_a_replaced_field_is_refused_under_its_class_name() -> None:
    order = Order("ORD-88", (), "warehouse-2")
    with pytest.raises(SerializationError, match=r"Order\.lines\[0\] is of type Decimal"):
        fields_data(order, lines=(Decimal("1"),))


@pytest.mark.parametrize("value", [{"sku": "SKU-1042"}, Order])
def test_fields_data_takes_a_dataclass_instance(value: Any) -> None:
    with pytest.raises(SerializationError, match="takes a dataclass instance"):
        fields_data(value)


def test_what_has_no_rule_is_refused_where_it_is() -> None:
    weighed = Product(**{**KETTLE.__dict__, "notes": {"tare": Decimal("0.15")}})
    with pytest.raises(
        SerializationError, match=r"value\.notes\['tare'\] is of type Decimal"
    ) as caught:
        as_data(weighed)
    assert caught.value.context == {"path": "value.notes['tare']", "type": "Decimal"}
    with pytest.raises(SerializationError, match=r"value\[1\] is of type Decimal"):
        as_data(["SKU-1042", Decimal("1")])


def test_keys_are_written_as_text() -> None:
    """A scalar key is written as ``json.dumps`` writes it, so both agree."""
    keys = {2024: "catalogue", None: "unfiled", True: "yes", False: "no", 1.5: "half"}
    data = as_data(keys)
    assert data == {
        "2024": "catalogue",
        "null": "unfiled",
        "true": "yes",
        "false": "no",
        "1.5": "half",
    }
    assert data == json.loads(json.dumps(keys))


def test_an_enum_key_is_written_as_its_value() -> None:
    assert as_data({Status.ACTIVE: 3}) == {"active": 3}


def test_keys_that_read_the_same_as_text_are_refused() -> None:
    """``1`` and ``"1"`` would both be written ``"1"``, and one would be lost."""
    shelves = Product(**{**KETTLE.__dict__, "notes": {"shelf": {1: "aisle 4", "1": "aisle 9"}}})
    with pytest.raises(
        SerializationError, match=r"value\.notes\['shelf'\] has two keys written '1': 1 and '1'"
    ):
        as_data(shelves)


def test_a_value_that_holds_itself_is_refused() -> None:
    catalogue: dict[str, Any] = {"sku": "SKU-1042"}
    catalogue["related"] = [catalogue]
    with pytest.raises(
        SerializationError, match=r"value\['related'\]\[0\] refers back to a value that contains it"
    ):
        as_data(catalogue)


def test_a_value_shared_by_two_fields_is_not_a_cycle() -> None:
    assert as_data({"a": HARBOR, "b": HARBOR}) == {
        "a": {"supplier_id": "S-07", "name": "Harbor Metalworks"},
        "b": {"supplier_id": "S-07", "name": "Harbor Metalworks"},
    }


def test_a_type_is_shown_its_way_at_every_depth() -> None:
    """Only the value a serialiser returned skips it, not what that value holds."""

    @dataclass(frozen=True)
    class Category:
        name: str
        children: tuple[Category, ...]
        internal_code: str = "hidden"

    def public(category: Category) -> dict[str, Any]:
        return {"name": category.name, "children": category.children}

    tree = Category("kitchen", (Category("kettles", ()),))
    assert as_data(tree, using={Category: public}) == {
        "name": "kitchen",
        "children": [{"name": "kettles", "children": []}],
    }


def test_a_serialiser_that_returns_what_it_was_given_is_walked_once() -> None:
    assert as_data(HARBOR, using={Supplier: lambda supplier: supplier}) == {
        "supplier_id": "S-07",
        "name": "Harbor Metalworks",
    }


def test_a_serialiser_that_wraps_its_value_is_a_cycle() -> None:
    with pytest.raises(SerializationError, match=r"value\['s'\] refers back"):
        as_data(HARBOR, using={Supplier: lambda supplier: {"s": supplier}})


def test_a_using_key_must_be_a_type() -> None:
    with pytest.raises(SerializationError, match="using takes types as keys"):
        as_data(HARBOR, using={list[int]: str})  # type: ignore[dict-item]


def test_a_refusal_inside_a_serialiser_names_the_outer_path() -> None:
    def strict(supplier: Supplier) -> Any:
        return as_data(Decimal("1"))

    with pytest.raises(
        SerializationError,
        match=r"value\['k'\]: the serialiser for Supplier raised SerializationError",
    ) as caught:
        as_data({"k": HARBOR}, using={Supplier: strict})
    assert caught.value.context["path"] == "value['k']"
    assert isinstance(caught.value.__cause__, SerializationError)


def test_a_to_dict_that_returns_its_own_value_is_refused() -> None:
    class Loop:
        def to_dict(self) -> Any:
            return self

    with pytest.raises(
        SerializationError, match=r"value: to_dict\(\) of Loop returned the value itself"
    ):
        as_data(Loop())


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_a_float_json_cannot_write_is_refused(number: float) -> None:
    with pytest.raises(
        SerializationError, match=r"value\.weight_kg is a non-finite float"
    ) as caught:
        as_data(Product(**{**KETTLE.__dict__, "weight_kg": number}))
    assert caught.value.context["path"] == "value.weight_kg"


def test_a_value_nested_deeply_is_walked() -> None:
    nested: list[Any] = []
    for _ in range(900):
        nested = [nested]
    assert json.dumps(as_data(nested)) == "[" * 901 + "]" * 901


def test_a_value_nested_past_what_the_walk_can_follow_is_refused() -> None:
    nested: list[Any] = []
    for _ in range(20_000):
        nested = [nested]
    with pytest.raises(SerializationError, match="nested too deeply"):
        as_data(nested)

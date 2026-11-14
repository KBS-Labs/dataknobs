"""A mapping that does not fit its class is refused by name.

``from_dict`` refuses a mapping in two ways: it leaves out a field the class
requires, or (under ``_UNKNOWN_KEYS = "raise"``) it carries a key the class
does not read. Both are the same authoring error, so both raise one type,
:class:`~dataknobs_common.exceptions.ConfigFieldError`, and neither reaches
the dataclass constructor's ``TypeError``.

The type is both a ``ValidationError`` and a ``ValueError``. The first is what
every door that documents a dataknobs refusal promises. The second keeps every
``except ValueError`` written against the unknown-key refusal catching it.

The messages name only things the code supplies (the class, its field names,
the keys the caller wrote) and never a value, so a caller may show one to the
author of the config without vetting it first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import pytest

from dataknobs_common import ConfigFieldError as ExportedConfigFieldError
from dataknobs_common.exceptions import (
    ConfigFieldError,
    DataknobsError,
    ValidationError,
)
from dataknobs_common.structured_config import StructuredConfig, StructuredConfigConsumer


@dataclass(frozen=True)
class _TwoRequired(StructuredConfig):
    name: str
    secret: str
    retries: int = 3
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Inner(StructuredConfig):
    url: str


@dataclass(frozen=True)
class _Outer(StructuredConfig):
    inner: _Inner | None = None


@dataclass(frozen=True)
class _Strict(StructuredConfig):
    host: str = "localhost"

    _UNKNOWN_KEYS: ClassVar[Literal["ignore", "raise"]] = "raise"


@dataclass(frozen=True)
class _Aliased(StructuredConfig):
    """A required field reachable through an alias the normalizer translates."""

    host: str

    @classmethod
    def _normalize_dict(cls, raw: dict[str, Any]) -> dict[str, Any]:
        if "hostname" in raw:
            raw.setdefault("host", raw.pop("hostname"))
        return raw


@dataclass(frozen=True)
class _NullableRequired(StructuredConfig):
    """Required, and legitimately ``None``: the base cannot read null as absent."""

    parent: str | None


@dataclass(frozen=True)
class _StrictRequired(StructuredConfig):
    """Closed and required: the refusal that runs first is the one reported."""

    host: str

    _UNKNOWN_KEYS: ClassVar[Literal["ignore", "raise"]] = "raise"


@dataclass(frozen=True)
class _DerivedRequired(_TwoRequired):
    """Inherits two required fields and adds one."""

    region: str = "eu"
    zone: str = field(kw_only=True)


@dataclass(frozen=True)
class _WithComputed(StructuredConfig):
    """A field outside ``__init__`` is never something a mapping must supply."""

    name: str
    derived: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "derived", self.name.upper())


class _Consumer(StructuredConfigConsumer[_TwoRequired]):
    CONFIG_CLS: ClassVar[type[_TwoRequired]] = _TwoRequired


class TestTheType:
    def test_it_is_both_a_validation_error_and_a_value_error(self) -> None:
        assert issubclass(ConfigFieldError, ValidationError)
        assert issubclass(ConfigFieldError, ValueError)
        assert issubclass(ConfigFieldError, DataknobsError)

    def test_it_constructs_through_the_mro_with_its_context(self) -> None:
        """``DataknobsError.__init__``'s ``super()`` resolves to ``ValueError``'s."""
        error = ConfigFieldError("m", context={"config_class": "X"})
        assert str(error) == "m"
        assert error.args == ("m",)
        assert error.context == {"config_class": "X"}
        assert error.details is error.context

    def test_it_is_exported_from_the_package(self) -> None:
        assert ExportedConfigFieldError is ConfigFieldError


class TestAMissingField:
    def test_it_raises_the_config_field_error_not_the_constructors_type_error(self) -> None:
        with pytest.raises(ConfigFieldError):
            _TwoRequired.from_dict({})

    def test_every_missing_field_is_named_at_once(self) -> None:
        with pytest.raises(ConfigFieldError) as excinfo:
            _TwoRequired.from_dict({"retries": 1})
        message = str(excinfo.value)
        assert "_TwoRequired" in message
        assert "'name'" in message and "'secret'" in message
        assert excinfo.value.context["config_class"] == "_TwoRequired"
        assert excinfo.value.context["missing"] == ["name", "secret"]

    def test_a_field_with_a_default_is_never_reported(self) -> None:
        with pytest.raises(ConfigFieldError) as excinfo:
            _TwoRequired.from_dict({"name": "n"})
        assert excinfo.value.context["missing"] == ["secret"]
        assert "retries" not in str(excinfo.value)
        assert "tags" not in str(excinfo.value)

    def test_no_value_the_caller_wrote_reaches_the_message(self) -> None:
        with pytest.raises(ConfigFieldError) as excinfo:
            _TwoRequired.from_dict({"name": "hunter2-not-a-name", "retries": 99})
        rendered = f"{excinfo.value} {excinfo.value.context}"
        assert "hunter2" not in rendered
        assert "99" not in rendered

    def test_a_field_supplied_through_an_alias_is_not_missing(self) -> None:
        """The check runs after ``_normalize_dict``, where an alias becomes a field."""
        assert _Aliased.from_dict({"hostname": "h"}).host == "h"

    def test_a_required_field_written_as_null_is_not_missing_at_the_base(self) -> None:
        assert _NullableRequired.from_dict({"parent": None}).parent is None

    def test_inherited_and_keyword_only_required_fields_are_named(self) -> None:
        with pytest.raises(ConfigFieldError) as excinfo:
            _DerivedRequired.from_dict({"name": "n"})
        assert excinfo.value.context["missing"] == ["secret", "zone"]

    def test_a_field_outside_init_is_never_required(self) -> None:
        assert _WithComputed.from_dict({"name": "n"}).derived == "N"
        with pytest.raises(ConfigFieldError) as excinfo:
            _WithComputed.from_dict({})
        assert excinfo.value.context["missing"] == ["name"]

    def test_a_misspelt_required_field_of_a_closed_class_is_reported_as_the_misspelling(
        self,
    ) -> None:
        """The unknown-key check runs first, so the author sees the near miss."""
        with pytest.raises(ConfigFieldError, match="did you mean 'host'") as excinfo:
            _StrictRequired.from_dict({"hots": "h"})
        assert "missing" not in excinfo.value.context
        assert excinfo.value.context["unknown"] == ["hots"]

    def test_a_missing_field_in_a_nested_section_names_the_nested_class(self) -> None:
        with pytest.raises(ConfigFieldError) as excinfo:
            _Outer.from_dict({"inner": {}})
        assert "_Inner" in str(excinfo.value)
        assert excinfo.value.context["missing"] == ["url"]


class TestAnUnknownKey:
    def test_it_raises_the_same_type_with_its_message_unchanged(self) -> None:
        with pytest.raises(ConfigFieldError) as excinfo:
            _Strict.from_dict({"hosst": "h"})
        assert str(excinfo.value) == (
            "_Strict does not accept 'hosst' (did you mean 'host'?). Accepted keys: host."
        )
        assert excinfo.value.context["unknown"] == ["hosst"]
        assert excinfo.value.context["accepted"] == ["host"]

    def test_an_except_value_error_still_catches_it(self) -> None:
        with pytest.raises(ValueError, match="hosst"):
            _Strict.from_dict({"hosst": "h"})


class TestFromComponentsWithNoConfig:
    def test_a_class_with_required_fields_is_refused_naming_them(self) -> None:
        with pytest.raises(ConfigFieldError) as excinfo:
            _Consumer.from_components()
        message = str(excinfo.value)
        assert "cannot be default-constructed" in message
        assert "'name'" in message and "'secret'" in message
        assert excinfo.value.context["missing"] == ["name", "secret"]

"""A refusal raised inside a block says where its configuration came from.

A component refuses its own configuration without knowing who configured it:
a database backend built for an ontology binding, or for a bot's grounded
source, names its table and nothing a reader of the ontology document or the
bot config would recognise. :func:`naming_refusals` is the one way the caller
that does know says so.
"""

from __future__ import annotations

import pytest

from dataknobs_common.exceptions import (
    ConfigurationError,
    DottedPathError,
    DottedPathReason,
    OperationError,
    ValidationError,
    naming_refusals,
)


def test_a_validation_refusal_is_prefixed_and_keeps_its_kind() -> None:
    original = ValidationError("field 'id' is not declared", context={"field": "id"})
    with pytest.raises(ValidationError) as caught:
        with naming_refusals("source 'cases'", context={"source": "cases"}):
            raise original

    assert str(caught.value) == "source 'cases': field 'id' is not declared"
    assert caught.value.context == {"field": "id", "source": "cases"}
    assert caught.value.__cause__ is original


def test_a_configuration_refusal_keeps_its_kind() -> None:
    with pytest.raises(ConfigurationError) as caught:
        with naming_refusals("ontology 'helpdesk', binding 'categories'"):
            raise ConfigurationError("`ensure_database: true` under `layout: native`")

    assert not isinstance(caught.value, ValidationError)
    assert str(caught.value).startswith("ontology 'helpdesk', binding 'categories': ")


def test_the_callers_context_wins_over_the_refusals() -> None:
    """The caller is the one who knows which source it was."""
    with pytest.raises(ValidationError) as caught:
        with naming_refusals("source 'b'", context={"source": "b"}):
            raise ValidationError("x", context={"source": "a", "table": "t"})

    assert caught.value.context == {"source": "b", "table": "t"}


def test_a_subclass_is_raised_as_its_kind() -> None:
    """A subclass may construct itself from other arguments, so its kind is what is raised.

    The original, with its own type and attributes, is the cause.
    """
    original = DottedPathError(
        "no attribute 'b'", ref="a.b", reason=DottedPathReason.ATTRIBUTE_NOT_FOUND
    )
    with pytest.raises(ConfigurationError) as caught:
        with naming_refusals("source 'cases'"):
            raise original

    assert type(caught.value) is ConfigurationError
    assert caught.value.__cause__ is original


@pytest.mark.parametrize("error", [ValueError("bad key"), OperationError("read-only")])
def test_anything_but_a_refusal_passes_through_untouched(error: Exception) -> None:
    with pytest.raises(type(error)) as caught:
        with naming_refusals("source 'cases'"):
            raise error

    assert caught.value is error


def test_a_block_that_raises_nothing_is_unaffected() -> None:
    with naming_refusals("source 'cases'"):
        value = 1
    assert value == 1

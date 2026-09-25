"""A database source reads two things from a field's metadata, and checks both.

``DatabaseSource.get_schema()`` turns a field's ``metadata["description"]`` and
``metadata["enum"]`` into the filter schema an extraction model is shown. The
schema reader checks the ``enum:`` a declaration writes, but a schema also
reaches a source built by hand, with its metadata set directly. A string enum
reached the filter schema as one allowed value per letter, and a non-string
description reached it as a JSON schema ``description`` that is not a string.
The source is the one reader of both, so it is the one place that can check
them on every route.

It also names the keys a field declaration takes for it
(:data:`SOURCE_FIELD_KEYS`), so a door that builds one refuses a key the source
would load and never read.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_common.exceptions import ValidationError

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.database import extract_schema_from_config
from dataknobs_data.fields import FieldType
from dataknobs_data.schema import DatabaseSchema, FieldSchema
from dataknobs_data.sources.database import SOURCE_FIELD_KEYS, DatabaseSource


def _source(**metadata: Any) -> DatabaseSource:
    schema = DatabaseSchema(
        fields={"dept": FieldSchema(name="dept", type=FieldType.STRING, metadata=metadata)}
    )
    return DatabaseSource(db=AsyncMemoryDatabase(), schema=schema, name="courses")


@pytest.mark.parametrize("enum", ["CS", 5, {"CS": 1}])
def test_a_hand_built_enum_that_is_not_a_list_is_refused(enum: Any) -> None:
    """It used to reach the filter schema as `list(enum)`: `['C', 'S']` for `"CS"`."""
    with pytest.raises(ValidationError) as excinfo:
        _source(enum=enum)
    message = str(excinfo.value)
    assert message.startswith("source 'courses': ")
    assert "'dept'" in message and "enum" in message
    assert excinfo.value.context["source"] == "courses"


@pytest.mark.parametrize("description", [5, ["a"], {"text": "x"}])
def test_a_description_that_is_not_a_string_is_refused(description: Any) -> None:
    """It used to reach the filter schema as a JSON schema `description` of another type."""
    with pytest.raises(ValidationError) as excinfo:
        _source(description=description)
    message = str(excinfo.value)
    assert message.startswith("source 'courses': ")
    assert "'dept'" in message and "description" in message


def test_a_valid_description_and_enum_still_reach_the_filter_schema() -> None:
    """The guard refuses only what the filter schema cannot carry."""
    dept = _source(description="The department", enum=("CS", "Math")).get_schema().fields["dept"]
    assert dept["description"] == "The department"
    assert dept["enum"] == ["CS", "Math"]


def test_the_keys_a_source_field_takes_are_the_ones_it_reads() -> None:
    """`get_schema()` reads a field's name, type and metadata (`enum:` folds into it)."""
    assert frozenset({"name", "type", "metadata", "enum"}) == SOURCE_FIELD_KEYS


@pytest.mark.parametrize("key", ["required", "default", "dimensions", "source_field"])
def test_a_key_the_source_never_reads_is_refused_through_its_key_set(key: str) -> None:
    """Each loads through the general key set and is discarded by the source."""
    with pytest.raises(ValidationError, match=rf"\['{key}'\]"):
        extract_schema_from_config(
            {"fields": {"dept": {"type": "string", key: True}}}, keys=SOURCE_FIELD_KEYS
        )

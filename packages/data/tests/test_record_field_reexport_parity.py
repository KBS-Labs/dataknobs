"""``Record``, ``Field`` and ``FieldType`` resolve to one object at every depth.

They are defined in ``dataknobs_common`` and reached through ``dataknobs_data``
by re-export. A package-level re-export alone would not be enough: thirty import
lines in this repository name the *module* path (``dataknobs_data.records``,
``dataknobs_data.fields``) rather than the package, and seven of them are
outside this package. So both depths are asserted, and by identity rather than
equality — an accidental second class would compare equal on a dataclass and
still break every ``isinstance`` downstream of it.
"""

from __future__ import annotations

import pytest

import dataknobs_common
import dataknobs_data
from dataknobs_common.fields import Field as CommonModuleField
from dataknobs_common.fields import FieldType as CommonModuleFieldType
from dataknobs_common.records import Record as CommonModuleRecord
from dataknobs_data.fields import Field as DataModuleField
from dataknobs_data.fields import FieldType as DataModuleFieldType
from dataknobs_data.records import Record as DataModuleRecord

np = pytest.importorskip("numpy")


def test_record_is_one_class_at_every_depth() -> None:
    assert dataknobs_data.Record is DataModuleRecord
    assert dataknobs_data.Record is CommonModuleRecord
    assert dataknobs_data.Record is dataknobs_common.Record


def test_field_is_one_class_at_every_depth() -> None:
    assert dataknobs_data.Field is DataModuleField
    assert dataknobs_data.Field is CommonModuleField
    assert dataknobs_data.Field is dataknobs_common.Field


def test_field_type_is_one_enum_at_every_depth() -> None:
    assert dataknobs_data.FieldType is DataModuleFieldType
    assert dataknobs_data.FieldType is CommonModuleFieldType
    assert dataknobs_data.FieldType is dataknobs_common.FieldType


def test_vector_field_is_only_reachable_from_this_package() -> None:
    """It needs numpy, so it stays here and is not re-exported upward."""
    assert not hasattr(dataknobs_common, "VectorField")
    assert issubclass(dataknobs_data.VectorField, dataknobs_common.Field)


def test_importing_this_package_registers_the_vector_types() -> None:
    """The other half of the pair asserted in the common package's suite.

    There, a vector payload builds a plain ``Field`` because nothing has
    registered the type. Here it builds a ``VectorField``, and the only
    difference between the two runs is that this package was imported.
    """
    built = dataknobs_data.Field.from_dict(
        {
            "name": "embedding",
            "value": [0.1, 0.2, 0.3],
            "type": "vector",
            "metadata": {"dimensions": 3},
        }
    )

    assert isinstance(built, dataknobs_data.VectorField)
    assert built.dimensions == 3

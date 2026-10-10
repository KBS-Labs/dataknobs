"""Every way a database's schema changes goes through ``set_schema``.

``set_schema``, ``add_field_schema`` and ``with_schema`` are three doors to one
change. A backend whose state derives from the schema -- the Postgres column
layout reads the declared fields -- had to override all three, and each
override restated what the base would produce before calling it. With the two
convenience doors built on ``set_schema``, a backend overrides one method, and
one that refuses the new schema leaves the old one in place, whichever door
it came through.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.fields import FieldType
from dataknobs_data.schema import DatabaseSchema, FieldSchema


class _Watching:
    seen: list[list[str]]
    refuse: bool = False

    def set_schema(self, schema: DatabaseSchema) -> None:
        self.seen.append(sorted(schema.fields))
        if self.refuse:
            raise ValueError("refused")
        super().set_schema(schema)  # type: ignore[misc]


class WatchingSync(_Watching, SyncMemoryDatabase):
    pass


class WatchingAsync(_Watching, AsyncMemoryDatabase):
    pass


def _db(cls: type) -> Any:
    db = cls()
    db.seen = []
    db.set_schema(DatabaseSchema.create(name=FieldType.STRING))
    db.seen.clear()
    return db


@pytest.mark.parametrize("cls", [WatchingSync, WatchingAsync], ids=lambda c: c.__name__)
def test_adding_a_field_goes_through_set_schema(cls: type) -> None:
    db = _db(cls)
    db.add_field_schema(FieldSchema("size", FieldType.INTEGER))
    assert db.seen == [["name", "size"]]
    assert sorted(db.schema.fields) == ["name", "size"]


@pytest.mark.parametrize("cls", [WatchingSync, WatchingAsync], ids=lambda c: c.__name__)
def test_with_schema_goes_through_set_schema_and_chains(cls: type) -> None:
    db = _db(cls)
    assert db.with_schema(title=FieldType.STRING) is db
    assert db.seen == [["title"]]


@pytest.mark.parametrize("cls", [WatchingSync, WatchingAsync], ids=lambda c: c.__name__)
def test_a_refused_field_leaves_the_schema_as_it_was(cls: type) -> None:
    db = _db(cls)
    before = db.schema
    db.refuse = True
    with pytest.raises(ValueError, match="refused"):
        db.add_field_schema(FieldSchema("size", FieldType.INTEGER))
    assert db.schema is before
    assert sorted(db.schema.fields) == ["name"]

"""Record-identity strategy for the database function library.

Several database functions need to know a record's *identity* — the stable id
under which the row is stored — so they can detect duplicates
(:class:`~dataknobs_fsm.functions.library.database.DatabaseBulkInsert`'s
``on_duplicate``), re-commit idempotently
(:class:`~dataknobs_fsm.functions.library.database.BatchCommit`), or upsert
(:class:`~dataknobs_fsm.functions.library.database.DatabaseUpsert`).

Identity derivation is consumer-specific, so it is expressed through the small
:class:`RecordIdentity` protocol rather than a single hard-coded rule. The
library ships two reference implementations and a resolver that turns the
``key_columns=`` / ``id_fn=`` / ``identity=`` constructor sugar (shared by the
identity-bearing functions) into a single ``RecordIdentity``:

- :class:`KeyColumnsIdentity` — join named key columns with a collision-safe
  separator (the common case); the default.
- :class:`CallableIdentity` — wrap any ``row -> id`` callable (the escape hatch
  for hashing, encoding, single-column raw values, natural keys, …).

Consumers may also implement :class:`RecordIdentity` directly and pass it as
``identity=``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

from dataknobs_common.exceptions import ConfigurationError

#: ASCII unit separator (U+001F). Used as the default join separator for
#: :class:`KeyColumnsIdentity` because it cannot appear in normal field text,
#: so composite keys cannot collide the way a printable separator (``"_"``)
#: lets them: ``["a_b", "c"]`` and ``["a", "b_c"]`` join to *distinct* ids.
DEFAULT_KEY_SEP = "\x1f"


@runtime_checkable
class RecordIdentity(Protocol):
    """A strategy that derives a stable storage id from a record."""

    def derive(self, row: Mapping[str, Any]) -> str | None:
        """Return the row's storage id, or ``None`` for "let the store assign".

        ``None`` means the record has no caller-defined identity — the backend
        generates one on ``create``. A non-``None`` id is the key under which
        the row is upserted and against which duplicates are detected.
        """
        ...


class KeyColumnsIdentity:
    """Derive an id by joining the values of named key columns.

    Args:
        key_columns: Columns whose values form the unique key. An empty list
            means "no identity" — :meth:`derive` returns ``None``.
        sep: Separator joining the column values. Defaults to
            :data:`DEFAULT_KEY_SEP` (ASCII unit separator), which is
            collision-safe for composite keys. Override only when the storage
            id must follow a specific printable format.
    """

    def __init__(
        self, key_columns: list[str], sep: str = DEFAULT_KEY_SEP
    ) -> None:
        self.key_columns = list(key_columns)
        self.sep = sep

    def derive(self, row: Mapping[str, Any]) -> str | None:
        if not self.key_columns:
            return None
        return self.sep.join(str(row.get(col)) for col in self.key_columns)


class CallableIdentity:
    """Derive an id from an arbitrary ``row -> id`` callable.

    The escape hatch for any identity scheme the column-join default does not
    cover (hashing, base64 encoding, a single raw column value, a natural key
    from a non-key field). The callable may return ``None`` to defer id
    assignment to the backend.

    Args:
        id_fn: ``Callable[[Mapping[str, Any]], str | None]``.
    """

    def __init__(
        self, id_fn: Callable[[Mapping[str, Any]], str | None]
    ) -> None:
        self.id_fn = id_fn

    def derive(self, row: Mapping[str, Any]) -> str | None:
        return self.id_fn(row)


def resolve_identity(
    *,
    identity: RecordIdentity | None = None,
    key_columns: list[str] | None = None,
    id_fn: Callable[[Mapping[str, Any]], str | None] | None = None,
) -> RecordIdentity | None:
    """Resolve the ``identity`` / ``key_columns`` / ``id_fn`` sugar to one strategy.

    Exactly one (or none) of the three may be supplied. Returns ``None`` when
    none is supplied — the "no caller-defined identity" case (create-only /
    backend-assigned ids).

    Raises:
        ConfigurationError: If more than one of the three is supplied, or if a
            supplied ``identity`` does not satisfy the :class:`RecordIdentity`
            protocol.
    """
    supplied = [
        name
        for name, value in (
            ("identity", identity),
            ("key_columns", key_columns),
            ("id_fn", id_fn),
        )
        if value is not None
    ]
    if len(supplied) > 1:
        raise ConfigurationError(
            "Specify at most one of identity=, key_columns=, id_fn= "
            f"(got {', '.join(supplied)})"
        )

    if identity is not None:
        if not isinstance(identity, RecordIdentity):
            raise ConfigurationError(
                "identity= must implement the RecordIdentity protocol "
                "(a 'derive(row) -> str | None' method)"
            )
        return identity
    if id_fn is not None:
        return CallableIdentity(id_fn)
    if key_columns is not None:
        return KeyColumnsIdentity(key_columns)
    return None

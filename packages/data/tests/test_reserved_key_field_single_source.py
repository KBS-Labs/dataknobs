"""Single-source guard for the reserved storage-key field name.

The rule "the query/sort field name ``id`` is reserved and targets a record's
storage key, not a ``data`` value stored under the same name" is one policy. It
must live in exactly one place — ``query.py``'s :func:`is_storage_key_field` — so
every backend's filter/sort translation agrees on the reserved name by
construction.

This guard fails if any backend module re-opens the literal ``field == "id"``
inline instead of consulting the predicate. That inline comparison is precisely
the drift an offline test cannot otherwise catch until a service-gated backend
runs: independent copies of the same check fall out of sync silently.
"""

from __future__ import annotations

import re
from pathlib import Path

import dataknobs_data

_SRC = Path(dataknobs_data.__file__).parent

# Matches ``<...>field == "id"`` / ``field == 'id'`` (the reserved-name literal),
# whether the receiver is ``filter.field``, ``sort_spec.field`` or a bare
# ``field`` local. Does not match ``field.startswith("metadata.")`` (a distinct
# metadata-column policy) or ``field_name == RESERVED_KEY_FIELD`` (the predicate).
_LITERAL = re.compile(r"""\bfield\s*==\s*['"]id['"]""")


def test_no_backend_open_codes_the_reserved_field_literal() -> None:
    """The reserved storage-key field name lives only in ``query.py``.

    Any backend filter/sort translation must call ``is_storage_key_field()``,
    never compare ``field == "id"`` inline — that inline literal is exactly the
    drift a same-behaviour-different-copies design invites.
    """
    offenders: list[str] = []
    for path in _SRC.rglob("*.py"):
        if path.name == "query.py":  # the single source of truth
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if _LITERAL.search(line):
                offenders.append(f"{path.relative_to(_SRC)}:{lineno}")
    assert not offenders, (
        "Reserved-field literal re-opened; call is_storage_key_field() instead: "
        + ", ".join(sorted(offenders))
    )

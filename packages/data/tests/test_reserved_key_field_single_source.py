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

Scope of this guard: it pins the **literal-comparison** re-drift form (``field ==
"id"`` in either direction). The other id-resolution form — resolving the field
through ``record.get_value("id")`` on the in-memory ComplexQuery scan path, which
carries no literal for a grep to find — is covered instead by the behavioral
cross-backend parity tests in ``test_queryable_keys.py`` (a ComplexQuery ``id``
filter/sort must agree with the native-SQL backends). The two guards are
complementary: this one is structural, that one is behavioral.
"""

from __future__ import annotations

import re
from pathlib import Path

import dataknobs_data

_SRC = Path(dataknobs_data.__file__).parent

# Matches ``field == "id"`` / ``field == 'id'`` and the reversed ``"id" == field``
# (the reserved-name literal), whether the receiver is a dotted attribute
# (``filter.field``, ``sort_spec.field``, ``self.filter.field``) or a bare
# ``field`` local — the reversed alternative allows a ``(?:\w+\.)*`` attribute
# prefix so a dotted receiver is not silently missed. Does not match
# ``field.startswith("metadata.")`` (a distinct metadata-column policy) or
# ``field_name == RESERVED_KEY_FIELD`` (the predicate itself).
_LITERAL = re.compile(
    r"""\bfield\s*==\s*['"]id['"]|['"]id['"]\s*==\s*(?:\w+\.)*field\b"""
)


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


def test_literal_regex_catches_forward_and_reversed_dotted_forms() -> None:
    """Pin the guard regex's own coverage so it cannot silently under-match.

    The reversed alternative must fire on a dotted-attribute receiver
    (``"id" == filter.field``) — the actual call-site style — not only on a bare
    ``field`` local. It must also stay clear of the predicate itself and the
    distinct ``metadata.`` prefix policy, or the guard would false-positive.
    """
    should_match = [
        'if field == "id":',
        'if filter.field == "id":',
        'if sort_spec.field == "id":',
        'if self.filter.field == "id":',
        'if "id" == field:',
        'if "id" == filter.field:',
        'if "id" == sort_spec.field:',
        'if "id" == self.filter.field:',
    ]
    should_not_match = [
        "return field_name == RESERVED_KEY_FIELD",
        'if field.startswith("metadata."):',
        'if is_storage_key_field(filter.field):',
    ]
    for line in should_match:
        assert _LITERAL.search(line), f"guard regex should match: {line!r}"
    for line in should_not_match:
        assert not _LITERAL.search(line), f"guard regex should NOT match: {line!r}"

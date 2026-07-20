"""Single-source guard for the SQL ``metadata.`` field-prefix routing.

The rule "a field prefixed ``metadata.`` addresses the metadata JSONB column,
every other non-storage-key field addresses the data column" is one policy. It
lives in exactly one place — :func:`resolve_json_column_and_path` in
``sql_base.py`` — so the filter and sort translators cannot disagree on where a
field lives.

This guard fails if any module re-opens the ``field.startswith("metadata.")``
literal inline instead of consulting the helper. It is independent of the
reserved-storage-key guard: the two policies target different columns (metadata
vs storage key) and must not entangle.
"""

from __future__ import annotations

import re
from pathlib import Path

import dataknobs_data
from dataknobs_data.backends.sql_base import resolve_json_column_and_path

_SRC = Path(dataknobs_data.__file__).parent

_LITERAL = re.compile(r"""startswith\(\s*['"]metadata\.['"]\s*\)""")


def test_no_module_open_codes_the_metadata_prefix_literal() -> None:
    """The ``metadata.`` prefix routing lives only in ``sql_base.py``'s helper.

    Any SQL filter/sort translation must call ``resolve_json_column_and_path()``,
    never test ``field.startswith("metadata.")`` inline — independent copies of
    the same prefix check are exactly the drift the helper eliminates.
    """
    offenders: list[str] = []
    for path in _SRC.rglob("*.py"):
        if path.name == "sql_base.py":  # the single source of truth
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if _LITERAL.search(line):
                offenders.append(f"{path.relative_to(_SRC)}:{lineno}")
    assert not offenders, (
        "metadata.-prefix literal re-opened; call resolve_json_column_and_path() "
        "instead: " + ", ".join(sorted(offenders))
    )


def test_resolve_json_column_and_path_routes_metadata_and_data() -> None:
    """Route ``metadata.<path>`` to the metadata column and every other field to
    the data column, stripping the prefix for the nested path.
    """
    assert resolve_json_column_and_path("metadata.tenant_id") == ("metadata", "tenant_id")
    assert resolve_json_column_and_path("metadata.a.b") == ("metadata", "a.b")
    assert resolve_json_column_and_path("config.timeout") == ("data", "config.timeout")
    assert resolve_json_column_and_path("status") == ("data", "status")
    # A field merely containing (not prefixed by) "metadata" is a data field.
    assert resolve_json_column_and_path("metadata_id") == ("data", "metadata_id")

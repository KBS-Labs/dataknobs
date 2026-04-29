"""Unit tests for quote_ident() SQL identifier quoting."""

import pytest

from dataknobs_utils.sql_utils import quote_ident


def test_quote_ident_plain():
    assert quote_ident("public") == '"public"'


def test_quote_ident_mixed_case():
    assert quote_ident("MyTable") == '"MyTable"'


def test_quote_ident_reserved_word():
    assert quote_ident("user") == '"user"'


def test_quote_ident_internal_double_quote():
    assert quote_ident('weird"name') == '"weird""name"'


def test_quote_ident_multiple_internal_quotes():
    assert quote_ident('"already"') == '"""already"""'


def test_quote_ident_with_space():
    assert quote_ident("my table") == '"my table"'


def test_quote_ident_qualified_name_not_split():
    """Qualified names are NOT auto-split — caller's responsibility."""
    assert quote_ident("schema.table") == '"schema.table"'


def test_quote_ident_empty_raises():
    with pytest.raises(ValueError):
        quote_ident("")


def test_quote_ident_none_raises():
    with pytest.raises(ValueError):
        quote_ident(None)  # type: ignore[arg-type]


def test_quote_ident_sqlite_dialect():
    assert quote_ident("records", dialect="sqlite") == '"records"'


def test_quote_ident_duckdb_dialect():
    assert quote_ident("records", dialect="duckdb") == '"records"'


def test_quote_ident_idempotent_semantics():
    """Double-quoting a plain identifier is semantically equivalent in SQL."""
    name = quote_ident("records")
    assert name == '"records"'

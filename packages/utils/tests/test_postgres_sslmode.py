"""``sslmode`` plumbing for the psycopg2-based Postgres connector.

These tests verify that ``DotenvPostgresConnector``/``PostgresDB`` forward an
explicit ``sslmode`` into ``psycopg2.connect`` (and omit it otherwise so
libpq's own default applies). The actual TLS handshake needs an
SSL-configured server and is exercised by integration tests; here we only
assert the connect-kwarg plumbing, intercepting the psycopg2 boundary
(a third-party driver that opens a real socket — no DataKnobs construct
wraps it, so the boundary is patched).
"""

from __future__ import annotations

from typing import Any

import psycopg2

from dataknobs_utils.sql_utils import DotenvPostgresConnector, PostgresDB


class _StandInConnection:
    """What ``psycopg2.connect`` returns, for the little of it used here.

    A bare ``object()`` was the previous stand-in and is the one thing a real
    connection is not: ``object`` is the sole built-in with no ``__weakref__``
    slot, so it cannot be weak-referenced, while ``psycopg2.extensions.connection``
    can. That difference stopped being invisible once the connector began
    holding its open connections in a ``weakref.WeakSet`` — the stand-in failed
    where every real connection succeeds.
    """

    closed = 0


def _capture_connect(monkeypatch: Any) -> dict[str, Any]:
    """Patch ``psycopg2.connect`` to capture its kwargs instead of connecting."""
    captured: dict[str, Any] = {}

    def fake_connect(**kwargs: Any) -> _StandInConnection:
        captured.update(kwargs)
        return _StandInConnection()

    monkeypatch.setattr(psycopg2, "connect", fake_connect)
    return captured


class TestConnectorSslmode:
    def test_sslmode_omitted_by_default(self, monkeypatch: Any) -> None:
        captured = _capture_connect(monkeypatch)
        conn = DotenvPostgresConnector(host="h", db="d", user="u", pwd="p", port=5432)
        assert conn.sslmode is None
        conn.get_conn()
        assert "sslmode" not in captured

    def test_explicit_sslmode_forwarded(self, monkeypatch: Any) -> None:
        captured = _capture_connect(monkeypatch)
        conn = DotenvPostgresConnector(
            host="h", db="d", user="u", pwd="p", port=5432, sslmode="require"
        )
        assert conn.sslmode == "require"
        conn.get_conn()
        assert captured["sslmode"] == "require"


class TestPostgresDBSslmode:
    def test_passthrough_to_new_connector(self) -> None:
        db = PostgresDB(host="h", db="d", user="u", pwd="p", port=5432, sslmode="verify-full")
        assert db._connector.sslmode == "verify-full"

    def test_prebuilt_connector_keeps_its_own_sslmode(self) -> None:
        connector = DotenvPostgresConnector(
            host="h", db="d", user="u", pwd="p", port=5432, sslmode="require"
        )
        # ``sslmode`` arg is ignored when a connector instance is supplied.
        db = PostgresDB(connector, sslmode="disable")
        assert db._connector.sslmode == "require"

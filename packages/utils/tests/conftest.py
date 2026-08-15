from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Dict, List, Self

import psycopg2
import pytest
from psycopg2 import extensions


def resources_path(package: str) -> str:
    """Compose the path to a package's resources.
    :param package: The name of the resource's package (subdir).
    :param filename: The name of the file under the package.
    :return: The file path to the package's resources.
    """
    # For the monorepo structure, resources are directly in the tests folder
    return str(Path(__file__).parent / "resources")


def resource(package: str, filename: str) -> str:
    """Compose the path to a resource.
    :param package: The name of the resource's package (subdir).
    :param filename: The name of the file under the package.
    :return: The file path to the resource.
    """
    return str(Path(resources_path(package)) / filename)


def resource_as_text(package: str, filename: str) -> str:
    """Load a resource's contents as a single text string.
    :param package: The name of the resource's package (subdir).
    :param filename: The name of the file under the package.
    :return: The resource contents.
    """
    path = resource(package, filename)
    with open(path, encoding="utf-8") as infile:
        return infile.read()


def resource_as_list(
    package: str, filename: str, ignore_comments: str = "#", ignore_empties: str = True
) -> List[str]:
    """Read each file line and add to a list.
    :param package: The package (subdirectory)
    :param filename: The filename (within the package directory)
    :param ignore_comments: If non-null, skip lines beginning with this value
    :param ignore_empties: True to skip empty lines
    """
    result = list()
    path = resource(package, filename)
    with open(path, encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not ignore_empties or line:
                if not ignore_comments or not line.startswith(ignore_comments):
                    result.append(line)
    return result


def resource_as_json(package: str, filename: str) -> Dict:
    """Load a resource's contents as json
    :param package: The name of the resource's package (subdir).
    :param filename: The name of the file under the package.
    :return: The resource contents.
    """
    return json.loads(resource_as_text(package, filename))


TEST_JSON_001 = "test-001.json"
TEST_JSON_002 = "test-002.json"
TEST_JSON_003 = "test-003.json"


@pytest.fixture
def test_utils_dir() -> str:
    return resources_path("utils")


@pytest.fixture
def test_json_001() -> Dict:
    return resource_as_text("utils", TEST_JSON_001)


@pytest.fixture
def test_json_002() -> Dict:
    return resource_as_text("utils", TEST_JSON_002)


@pytest.fixture
def test_json_003() -> Dict:
    return resource_as_text("utils", TEST_JSON_003)


# --------------------------------------------------------------------------
# A stand-in for what ``psycopg2.connect`` returns.
#
# ``DotenvPostgresConnector._is_usable`` decides whether a pooled connection
# may be handed back, and every branch of that decision needs a connection in
# a specific state — idle, mid-transaction, aborted, or killed by the server.
# Reaching some of those against a real server means terminating a backend
# from a second session. The integration suite does exactly that, and it is
# skipped wherever no server is reachable, which is every CI job here.
#
# So the decision is pinned twice, at two different joints:
#
#   * here, in-process and always running — given a connection reporting
#     state X, does the ladder reach the right verdict?
#   * in tests/integration/test_postgres_connection_lifecycle.py, when a
#     server is reachable — does a *real* psycopg2 connection actually report
#     state X in the situation the ladder assumes it does?
#
# Neither is sufficient alone, and the split is the point: a stand-in can only
# be as truthful as what it was built from, so what it was built from is
# recorded rather than assumed. Measured against PostgreSQL 16 with
# psycopg2-binary 2.9, probing with ``SELECT 1``:
#
#   | situation                  | exception                  | closed after |
#   |----------------------------|----------------------------|--------------|
#   | aborted transaction, alive | InFailedSqlTransaction     | 0            |
#   | statement cancelled, alive | QueryCanceled              | 0            |
#   | backend terminated         | OperationalError           | 2            |
#   | terminated mid-query       | OperationalError           | 2            |
#   | already closed locally     | InterfaceError             | 1            |
#
# The load-bearing column is the last, not the exception type: psycopg2 marks
# the connection closed when the transport failed and leaves it open when the
# server merely refused the statement. That is why ``_is_usable`` consults
# ``conn.closed`` after a failed probe rather than classifying the exception —
# the connection's own verdict, rather than an inference about the error.
# ``error_is_transport`` selects a row of that table, and the stand-in mimics
# both consequences psycopg2 has in that case: ``closed`` becomes 2 and the
# transaction status becomes UNKNOWN.
# --------------------------------------------------------------------------


class StandInCursor:
    """The cursor surface the connector uses: a context manager and execute."""

    def __init__(self, conn: StandInConnection) -> None:
        self._conn = conn

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        """Record the statement, then fail the way the scenario says to."""
        self._conn.executed.append(sql)
        # Captured at execute time rather than after: whether the probe ran
        # under autocommit is the property, and the caller restores the prior
        # value on its way out, so reading it afterwards answers nothing.
        self._conn.autocommit_at_execute = self._conn.autocommit
        error = self._conn.probe_error
        if error is None:
            return
        if self._conn.error_is_transport:
            self._conn.closed = 2
            self._conn.status = extensions.TRANSACTION_STATUS_UNKNOWN
        raise error


class StandInConnection:
    """A psycopg2 connection, for the surface the connector actually touches.

    A bare ``object()`` was once the stand-in here and is the one thing a real
    connection is not: ``object`` is the sole built-in with no ``__weakref__``
    slot, so it cannot be weak-referenced, while
    ``psycopg2.extensions.connection`` can. That difference stopped being
    invisible once the connector began holding its open connections in a
    ``weakref.WeakSet``. An ordinary Python class is weak-referenceable, so
    this one is faithful where that matters.
    """

    def __init__(
        self,
        *,
        status: int = extensions.TRANSACTION_STATUS_IDLE,
        probe_error: BaseException | None = None,
        error_is_transport: bool = False,
        closed: int = 0,
        autocommit: bool = False,
        close_error: BaseException | None = None,
    ) -> None:
        self.status = status
        self.probe_error = probe_error
        self.error_is_transport = error_is_transport
        self.closed = closed
        self.autocommit = autocommit
        self.close_error = close_error
        self.executed: list[str] = []
        self.autocommit_at_execute: bool | None = None

    def get_transaction_status(self) -> int:
        return self.status

    def cursor(self) -> StandInCursor:
        return StandInCursor(self)

    def close(self) -> None:
        if self.close_error is not None:
            raise self.close_error
        self.closed = 1


@pytest.fixture
def stand_in() -> type[StandInConnection]:
    """The connection stand-in class, for tests to instantiate per scenario."""
    return StandInConnection


@pytest.fixture
def hand_out(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[Any]], None]:
    """Make ``psycopg2.connect`` yield the given connections, in order."""

    def install(conns: list[Any]) -> None:
        remaining = list(conns)

        def fake_connect(**_kwargs: Any) -> Any:
            return remaining.pop(0)

        monkeypatch.setattr(psycopg2, "connect", fake_connect)

    return install

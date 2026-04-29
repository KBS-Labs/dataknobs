"""Tests for ``dataknobs_common.testing.elasticsearch_fixtures``.

Covers:

* ``wait_for_elasticsearch`` retry + timeout paths (socket / RequestHelper
  stubbed to control failure / success ordering, ``time.sleep`` neutralized).
* ``elasticsearch_connection_params`` Docker-detection branch (env-var driven).
* Cleanup branch tightened to specific exceptions: ``ConnectionError`` /
  ``ValueError`` are logged and swallowed, other exceptions propagate.
"""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from dataknobs_common.testing import (
    elasticsearch_fixtures,
    wait_for_elasticsearch,
)

# -- wait_for_elasticsearch -----------------------------------------------


class _FakeSocket:
    """Patches socket.socket() to return a controllable connect_ex result."""

    instances: ClassVar[list[_FakeSocket]] = []

    def __init__(self, results_queue: list[int]) -> None:
        self._results = results_queue
        _FakeSocket.instances.append(self)

    def settimeout(self, _seconds: float) -> None:
        pass

    def connect_ex(self, _addr: tuple[str, int]) -> int:
        if not self._results:
            return 1  # closed by default once queue exhausted
        return self._results.pop(0)

    def close(self) -> None:
        pass


def _patch_sleep(monkeypatch) -> None:
    monkeypatch.setattr(
        "dataknobs_common.testing.elasticsearch_fixtures.time.sleep",
        lambda _seconds: None,
    )


def test_wait_for_elasticsearch_returns_true_on_yellow_status(monkeypatch):
    """Cluster status ``yellow`` is acceptable as ready."""
    _patch_sleep(monkeypatch)

    queue = [0]  # port open on first try

    def _fake_socket_factory(*_args, **_kwargs):
        return _FakeSocket(queue)

    monkeypatch.setattr(
        "dataknobs_common.testing.elasticsearch_fixtures.socket.socket",
        _fake_socket_factory,
    )

    class _FakeResponse:
        succeeded: ClassVar[bool] = True
        json: ClassVar[dict[str, Any]] = {"status": "yellow"}

    class _FakeRequestHelper:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def get(self, _path: str) -> _FakeResponse:
            return _FakeResponse()

    fake_requests = SimpleNamespace(
        exceptions=SimpleNamespace(
            ConnectionError=ConnectionError,
            Timeout=TimeoutError,
        ),
    )
    fake_requests_utils = SimpleNamespace(RequestHelper=_FakeRequestHelper)

    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    monkeypatch.setitem(
        sys.modules,
        "dataknobs_utils.requests_utils",
        fake_requests_utils,
    )

    assert wait_for_elasticsearch("h", 9200, max_retries=5) is True


def test_wait_for_elasticsearch_raises_when_port_never_opens(monkeypatch):
    """ConnectionError is raised after ``max_retries`` if the port never opens."""
    _patch_sleep(monkeypatch)

    queue: list[int] = []  # always returns 1 (closed)

    def _fake_socket_factory(*_args, **_kwargs):
        return _FakeSocket(queue)

    monkeypatch.setattr(
        "dataknobs_common.testing.elasticsearch_fixtures.socket.socket",
        _fake_socket_factory,
    )

    fake_requests = SimpleNamespace(
        exceptions=SimpleNamespace(
            ConnectionError=ConnectionError,
            Timeout=TimeoutError,
        ),
    )
    fake_requests_utils = SimpleNamespace(RequestHelper=lambda *a, **k: None)

    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    monkeypatch.setitem(
        sys.modules,
        "dataknobs_utils.requests_utils",
        fake_requests_utils,
    )

    with pytest.raises(ConnectionError):
        wait_for_elasticsearch("h", 9200, max_retries=3)


# -- elasticsearch_connection_params Docker detection ---------------------


def _call_params_fixture() -> dict[str, Any]:
    fixture_fn = elasticsearch_fixtures.elasticsearch_connection_params.__wrapped__  # type: ignore[attr-defined]
    return fixture_fn()


def _clear_es_env(monkeypatch) -> None:
    for name in (
        "DOCKER_CONTAINER",
        "ELASTICSEARCH_HOST",
        "ELASTICSEARCH_PORT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_elasticsearch_connection_params_localhost_default(monkeypatch):
    """Outside Docker, host defaults to ``localhost``."""
    _clear_es_env(monkeypatch)
    real_exists = os.path.exists
    monkeypatch.setattr(
        "dataknobs_common.testing.elasticsearch_fixtures.os.path.exists",
        lambda p: False if p == "/.dockerenv" else real_exists(p),
    )

    params = _call_params_fixture()
    assert params == {"host": "localhost", "port": 9200}


def test_elasticsearch_connection_params_docker_default(monkeypatch):
    """Inside Docker (DOCKER_CONTAINER set), host defaults to ``elasticsearch``."""
    _clear_es_env(monkeypatch)
    monkeypatch.setenv("DOCKER_CONTAINER", "1")
    monkeypatch.setattr(
        "dataknobs_common.testing.elasticsearch_fixtures.os.path.exists",
        lambda _p: False,
    )

    params = _call_params_fixture()
    assert params["host"] == "elasticsearch"


def test_elasticsearch_connection_params_env_overrides(monkeypatch):
    """Explicit env vars override Docker-detection defaults."""
    _clear_es_env(monkeypatch)
    monkeypatch.setenv("DOCKER_CONTAINER", "1")
    monkeypatch.setenv("ELASTICSEARCH_HOST", "es-host")
    monkeypatch.setenv("ELASTICSEARCH_PORT", "9999")

    params = _call_params_fixture()
    assert params == {"host": "es-host", "port": 9999}


# -- Cleanup branch exception specificity ---------------------------------


class _RecordingHandler(logging.Handler):
    """Captures log records emitted by the fixture's logger."""

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _drive_cleanup_with_exception(
    monkeypatch,
    exc_to_raise: type[BaseException],
) -> tuple[list[Any], BaseException | None]:
    """Drive the factory-fixture cleanup with a stubbed SimplifiedElasticsearchIndex.

    Returns (log_records, propagated_exception_or_none).
    """

    class _StubIndex:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def exists(self) -> bool:
            raise exc_to_raise("simulated cleanup failure")

        def delete(self) -> None:
            raise AssertionError("delete should not be reached")

    fake_module = SimpleNamespace(SimplifiedElasticsearchIndex=_StubIndex)
    monkeypatch.setitem(
        sys.modules, "dataknobs_utils.elasticsearch_utils", fake_module
    )

    handler = _RecordingHandler()
    logger = logging.getLogger(
        "dataknobs_common.testing.elasticsearch_fixtures"
    )
    prior_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        # Build the factory by calling the fixture function directly.
        params = {"host": "h", "port": 9200}
        factory_fixture = (
            elasticsearch_fixtures.make_elasticsearch_test_index.__wrapped__  # type: ignore[attr-defined]
        )
        factory = factory_fixture(None, params)
        gen = factory("test_records_")
        next(gen)  # enter the try, yield config

        propagated: BaseException | None = None
        try:
            # Drive teardown by closing the generator (triggers finally).
            gen.close()
        except BaseException as exc:
            propagated = exc

        return handler.records, propagated
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)


def test_cleanup_swallows_connection_error_and_logs(monkeypatch):
    """ConnectionError during cleanup is logged at WARNING and swallowed."""
    records, propagated = _drive_cleanup_with_exception(
        monkeypatch, ConnectionError
    )
    assert propagated is None
    assert any("Failed to clean up" in r.getMessage() for r in records)
    assert all(r.levelname == "WARNING" for r in records)


def test_cleanup_swallows_value_error_and_logs(monkeypatch):
    """ValueError during cleanup is logged at WARNING and swallowed."""
    records, propagated = _drive_cleanup_with_exception(monkeypatch, ValueError)
    assert propagated is None
    assert any("Failed to clean up" in r.getMessage() for r in records)


def test_cleanup_propagates_unexpected_exception(monkeypatch):
    """Non-(ConnectionError, ValueError) cleanup exceptions propagate."""
    _records, propagated = _drive_cleanup_with_exception(monkeypatch, RuntimeError)
    assert isinstance(propagated, RuntimeError)

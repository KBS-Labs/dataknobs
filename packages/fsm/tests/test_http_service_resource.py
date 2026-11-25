"""``HTTPServiceResource`` against a real local server.

Every request goes through ``urllib`` to a server on a random port.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError

import pytest

from dataknobs_fsm.functions.base import ResourceError
from dataknobs_fsm.resources.base import ResourceHealth, ResourceStatus
from dataknobs_fsm.resources.http import HTTPServiceResource, HTTPSession

#: What a server or a proxy puts in its status line is not this package's text.
_SERVER_WORDS = "upstream svc:hunter2@db.internal"


class _Server(ThreadingHTTPServer):
    """Answers each request with the next scripted status, then 200."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.statuses: list[int] = []
        self.requests: list[dict[str, Any]] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def _answer(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.server.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "headers": dict(self.headers),
                "body": self.rfile.read(length),
            }
        )
        status = self.server.statuses.pop(0) if self.server.statuses else 200
        body = b"not json" if self.path == "/text" else b'{"ok": true}'
        self.send_response(status, None if status == 200 else _SERVER_WORDS)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = _answer  # noqa: N815

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def server() -> Iterator[_Server]:
    running = _Server()
    thread = threading.Thread(target=running.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    yield running
    running.shutdown()
    running.server_close()
    thread.join()


@pytest.fixture
def api(server: _Server) -> HTTPServiceResource:
    return HTTPServiceResource("orders_api", server.url)


def _session(api: HTTPServiceResource, **kwargs: Any) -> HTTPSession:
    return api.acquire(retry_delay=0, **kwargs)


def _unused_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class TestRequests:
    @pytest.mark.parametrize("method", ["get", "post", "put", "delete"])
    def test_each_method_sends_itself_and_returns_the_decoded_body(self, api, server, method):
        response = getattr(api, method)("/orders")

        assert response["status"] == 200
        assert response["body"] == {"ok": True}
        assert (server.requests[0]["method"], server.requests[0]["path"]) == (
            method.upper(),
            "/orders",
        )

    def test_a_dict_is_sent_as_json(self, api, server):
        api.post("/orders", {"id": 1})
        assert json.loads(server.requests[0]["body"]) == {"id": 1}
        assert server.requests[0]["headers"]["Content-Type"] == "application/json"

    def test_bytes_are_sent_as_they_are(self, api, server):
        api.put("/orders", b"raw")
        assert server.requests[0]["body"] == b"raw"

    def test_a_body_that_is_not_json_is_returned_as_bytes(self, api):
        assert api.get("/text")["body"] == b"not json"

    def test_headers_for_one_request_are_added_to_the_sessions(self, server):
        api = HTTPServiceResource("a", server.url, headers={"X-Base": "b"})
        api.get("/", headers={"X-One": "o"})
        sent = server.requests[0]["headers"]
        assert (sent["X-Base"], sent["X-One"]) == ("b", "o")

    @pytest.mark.parametrize(
        ("auth", "header", "value"),
        [
            ({"type": "bearer", "token": "t"}, "Authorization", "Bearer t"),
            (
                {"type": "basic", "username": "u", "password": "p"},
                "Authorization",
                "Basic dTpw",
            ),
            ({"type": "api_key", "key_value": "k"}, "X-Api-Key", "k"),
            ({"type": "api_key", "key_name": "X-Mine", "key_value": "k"}, "X-Mine", "k"),
            ({"type": "api_key"}, "X-Api-Key", None),
        ],
    )
    def test_auth_is_sent(self, server, auth, header, value):
        HTTPServiceResource("a", server.url, auth=auth).get("/")
        assert server.requests[0]["headers"].get(header) == value

    def test_no_auth_sends_none(self, api, server):
        api.get("/")
        assert "Authorization" not in server.requests[0]["headers"]

    def test_auth_is_not_written_into_the_callers_headers(self, server):
        """Two resources given one dict must not send each other's credential."""
        shared = {"X-Base": "b"}
        HTTPServiceResource("a", server.url, headers=shared, auth={"type": "bearer", "token": "t"})
        HTTPServiceResource("b", server.url, headers=shared).get("/")

        assert shared == {"X-Base": "b"}
        assert "Authorization" not in server.requests[0]["headers"]
        assert server.requests[0]["headers"]["X-Base"] == "b"


class TestSessions:
    def test_a_released_session_leaves_the_resource_idle(self, api):
        session = api.acquire()
        assert api.status is ResourceStatus.ACTIVE
        api.release(session)
        assert api.status is ResourceStatus.IDLE

    def test_the_context_manager_releases(self, api):
        with api.session_context() as session:
            assert api.validate(session)
        assert api.status is ResourceStatus.IDLE

    def test_something_else_is_not_a_valid_session(self, api):
        assert api.validate(object()) is False

    def test_a_session_that_cannot_be_built_names_the_operation(self, api):
        with pytest.raises(ResourceError) as excinfo:
            api.acquire(headers=3)
        assert str(excinfo.value) == "Failed to acquire HTTP session (TypeError)"
        assert excinfo.value.operation == "acquire"
        assert isinstance(excinfo.value.__cause__, TypeError)
        assert api.status is ResourceStatus.ERROR

    def test_a_reachable_service_is_healthy(self, api):
        assert api.health_check() is ResourceHealth.HEALTHY

    def test_an_unreachable_service_is_unhealthy(self):
        api = HTTPServiceResource("a", f"http://127.0.0.1:{_unused_port()}", max_retries=1)
        assert api.health_check() is ResourceHealth.UNHEALTHY


class TestFailures:
    def test_a_client_error_is_not_retried_and_says_the_standard_phrase(self, api, server):
        server.statuses = [404]
        with pytest.raises(ResourceError) as excinfo:
            api.get("/orders/9", session=_session(api))

        assert str(excinfo.value) == "HTTP 404: Not Found"
        assert excinfo.value.operation == "GET /orders/9"
        assert excinfo.value.__cause__.reason == _SERVER_WORDS
        assert len(server.requests) == 1

    def test_a_code_with_no_standard_phrase_is_given_alone(self, api, server):
        server.statuses = [499]
        with pytest.raises(ResourceError, match=r"^HTTP 499$"):
            api.get("/", session=_session(api))

    def test_a_server_error_is_retried(self, api, server):
        server.statuses = [503, 503]
        assert api.get("/", session=_session(api))["status"] == 200
        assert len(server.requests) == 3

    def test_a_server_error_that_lasts_names_its_status(self, api, server):
        server.statuses = [503] * 3
        with pytest.raises(ResourceError) as excinfo:
            api.get("/", session=_session(api))

        assert str(excinfo.value) == (
            "Request failed after 3 retries (HTTP 503: Service Unavailable)"
        )
        assert isinstance(excinfo.value.__cause__, HTTPError)

    def test_a_refused_connection_is_named_by_its_type(self):
        api = HTTPServiceResource("a", f"http://127.0.0.1:{_unused_port()}")
        with pytest.raises(ResourceError) as excinfo:
            api.get("/", session=_session(api, max_retries=2))

        assert str(excinfo.value) == "Request failed after 2 retries (URLError)"
        assert "refused" in str(excinfo.value.__cause__)

    def test_a_credential_in_the_base_url_is_not_in_the_message(self):
        api = HTTPServiceResource("a", "http://svc:hunter2@127.0.0.1")
        with pytest.raises(ResourceError) as excinfo:
            api.get("/", session=_session(api, max_retries=1))

        assert str(excinfo.value) == "Request failed after 1 retries (InvalidURL)"
        assert "hunter2" in str(excinfo.value.__cause__)

    def test_a_session_allowed_no_attempts_fails_without_a_cause(self, api, server):
        with pytest.raises(ResourceError) as excinfo:
            api.get("/", session=_session(api, max_retries=0))
        assert str(excinfo.value) == "Request failed after 0 retries"
        assert excinfo.value.__cause__ is None
        assert server.requests == []


class TestCircuit:
    def _opened(self, api: HTTPServiceResource, server: _Server) -> HTTPSession:
        session = _session(api)
        server.statuses = [404] * session.failure_threshold
        for _ in range(session.failure_threshold):
            with pytest.raises(ResourceError, match="HTTP 404"):
                api.get("/", session=session)
        return session

    def test_an_open_circuit_refuses_without_a_request(self, api, server):
        session = self._opened(api, server)
        sent = len(server.requests)

        with pytest.raises(ResourceError) as excinfo:
            api.get("/orders", session=session)

        assert str(excinfo.value) == "Circuit breaker is open - service unavailable"
        assert excinfo.value.resource_name == "orders_api"
        assert excinfo.value.operation == "GET /orders"
        assert len(server.requests) == sent
        assert api.validate(session) is False

    def test_a_circuit_tries_again_after_its_wait_and_closes_on_success(self, api, server):
        session = self._opened(api, server)
        session.last_failure_time = time.time() - session.circuit_half_open_after - 1

        assert api.get("/", session=session)["status"] == 200
        assert (session.circuit_open, session.failure_count) == (False, 0)

    def test_a_failure_after_the_wait_opens_it_again(self, api, server):
        session = self._opened(api, server)
        session.last_failure_time = time.time() - session.circuit_half_open_after - 1
        server.statuses = [404]

        with pytest.raises(ResourceError, match="HTTP 404"):
            api.get("/", session=session)

        assert session.circuit_open is True

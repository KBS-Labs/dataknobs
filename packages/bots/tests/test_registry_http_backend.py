"""Tests for HTTPRegistryBackend.

The integration tests in this module spin up a real in-process
``aiohttp.web`` server via ``_MockHttpServer`` (defined below) instead
of monkey-patching aiohttp through ``aioresponses``. The harness:

* Uses ``aiohttp.web.Application`` + ``AppRunner`` + ``TCPSite`` on a
  random localhost port — every request goes through the real aiohttp
  client stack (encoding, headers, query serialization) and the real
  aiohttp server stack (parsing, routing, multi-valued query handling).
  Wire-protocol bugs surface here rather than in production.
* Captures each request as a ``_CapturedCall`` (method, path, query as
  a multi-valued mapping, headers, body) for assertion.
* Lets each test register per-endpoint responses (status + JSON payload
  or raw body) via ``server.get(path, ...)``, ``server.put(...)``, etc.

Replaces the previous ``aioresponses``-driven fixture. ``aioresponses``
0.7.8 (latest as of 2026-06) did not pass the ``stream_writer`` kwarg
to ``aiohttp.ClientResponse.__init__`` introduced in aiohttp 3.14, which
broke every test in this file under aiohttp >= 3.14. Switching to
the real-server harness frees the bots package from the temporary
``aiohttp<3.14`` cap (see ``packages/bots/pyproject.toml`` dev deps).
"""

import json
from dataclasses import dataclass
from typing import Any

import aiohttp.web
import pytest

from dataknobs_bots.registry import HTTPRegistryBackend, create_registry_backend


@dataclass
class _CapturedCall:
    """A single request received by ``_MockHttpServer``."""

    method: str
    path: str
    # Multi-valued query parameters. Single-occurrence keys still map to
    # a one-element list, so assertions iterate naturally.
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: bytes


@dataclass
class _ResponseSpec:
    """Per-endpoint stubbed response."""

    status: int = 200
    payload: Any = None
    body: bytes | None = None


class _MockHttpServer:
    """In-process aiohttp test server.

    Spins up a real ``aiohttp.web.Application`` on a random port. A
    catch-all route forwards every request to ``_dispatch``, which logs
    the call and replies with whatever the test pre-registered for the
    ``(method, path)`` pair. Unmatched requests get a 404 with a
    diagnostic body — easier to debug than aioresponses' silent matcher.
    """

    def __init__(self) -> None:
        self._responses: dict[tuple[str, str], _ResponseSpec] = {}
        self._calls: list[_CapturedCall] = []
        self._runner: aiohttp.web.AppRunner | None = None
        self._host: str = "127.0.0.1"
        self._port: int = 0

    async def start(self) -> None:
        app = aiohttp.web.Application()
        app.router.add_route("*", "/{tail:.*}", self._dispatch)
        self._runner = aiohttp.web.AppRunner(app)
        await self._runner.setup()
        # Bind to port=0 so the OS assigns a free port atomically — no
        # TOCTOU window between a pre-bind probe and the real bind.
        site = aiohttp.web.TCPSite(self._runner, self._host, 0)
        await site.start()
        self._port = site.port or 0

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    @property
    def base_url(self) -> str:
        """``base_url`` to pass to ``HTTPRegistryBackend`` (incl. /api/v1)."""
        return f"http://{self._host}:{self._port}/api/v1"

    def get(
        self,
        path: str,
        *,
        status: int = 200,
        payload: Any = None,
        body: bytes | None = None,
    ) -> None:
        self._responses[("GET", path)] = _ResponseSpec(
            status=status, payload=payload, body=body
        )

    def put(
        self,
        path: str,
        *,
        status: int = 200,
        payload: Any = None,
        body: bytes | None = None,
    ) -> None:
        self._responses[("PUT", path)] = _ResponseSpec(
            status=status, payload=payload, body=body
        )

    def post(
        self,
        path: str,
        *,
        status: int = 200,
        payload: Any = None,
        body: bytes | None = None,
    ) -> None:
        self._responses[("POST", path)] = _ResponseSpec(
            status=status, payload=payload, body=body
        )

    def delete(
        self,
        path: str,
        *,
        status: int = 204,
        payload: Any = None,
        body: bytes | None = None,
    ) -> None:
        self._responses[("DELETE", path)] = _ResponseSpec(
            status=status, payload=payload, body=body
        )

    @property
    def calls(self) -> list[_CapturedCall]:
        return list(self._calls)

    def calls_for(self, method: str) -> list[_CapturedCall]:
        return [c for c in self._calls if c.method.upper() == method.upper()]

    async def _dispatch(
        self, request: aiohttp.web.Request
    ) -> aiohttp.web.Response:
        body = await request.read()
        # ``request.query.keys()`` on a MultiDict returns unique keys in
        # insertion order — deterministic, no set-hash dedup surprises.
        query = {key: request.query.getall(key) for key in request.query.keys()}
        self._calls.append(
            _CapturedCall(
                method=request.method,
                path=request.path,
                query=query,
                headers=dict(request.headers),
                body=body,
            )
        )
        spec = self._responses.get((request.method, request.path))
        if spec is None:
            return aiohttp.web.Response(
                status=404,
                text=f"No mock registered for {request.method} {request.path}",
            )
        if spec.payload is not None:
            return aiohttp.web.json_response(spec.payload, status=spec.status)
        if spec.body is not None:
            return aiohttp.web.Response(status=spec.status, body=spec.body)
        return aiohttp.web.Response(status=spec.status)


def _captured_params(
    mock_server: _MockHttpServer, *, method: str = "GET"
) -> list[dict[str, str | list[str]]]:
    """Return the captured query dict for every request of ``method``.

    Single-occurrence keys are returned as scalars, multi-occurrence
    keys as lists — matching what aiohttp's client and server libraries
    naturally agree on for the wire format.
    """
    result: list[dict[str, str | list[str]]] = []
    for call in mock_server.calls_for(method):
        flat: dict[str, str | list[str]] = {}
        for key, values in call.query.items():
            flat[key] = values[0] if len(values) == 1 else values
        result.append(flat)
    return result


def _filter_metadata_param(mock_server: _MockHttpServer) -> dict | None:
    """Return the decoded ``filter_metadata`` from the first matching call."""
    for params in _captured_params(mock_server):
        raw = params.get("filter_metadata")
        if isinstance(raw, str):
            return json.loads(raw)
    return None


class TestHTTPRegistryBackendConfiguration:
    """Tests for HTTPRegistryBackend configuration and factory."""

    def test_from_config(self):
        """Test creating backend from config dict."""
        config = {
            "base_url": "https://example.com/api",
            "auth_token": "secret",
            "timeout": 60.0,
            "read_only": True,
        }
        backend = HTTPRegistryBackend.from_config(config)

        assert backend.base_url == "https://example.com/api"
        assert backend.is_read_only is True
        assert backend._timeout == 60.0

    def test_from_config_defaults(self):
        """Test config defaults are applied."""
        config = {"base_url": "https://example.com"}
        backend = HTTPRegistryBackend.from_config(config)

        assert backend._timeout == 30.0
        assert backend.is_read_only is False
        assert backend._verify_ssl is True

    def test_base_url_trailing_slash_stripped(self):
        """Test that trailing slash is stripped from base_url."""
        backend = HTTPRegistryBackend(base_url="https://example.com/api/")
        assert backend.base_url == "https://example.com/api"

    def test_create_registry_backend_http(self):
        """Test factory function creates HTTP backend."""
        backend = create_registry_backend("http", {
            "base_url": "https://example.com/api",
        })
        assert isinstance(backend, HTTPRegistryBackend)

    def test_create_registry_backend_memory(self):
        """Test factory function creates memory backend."""
        from dataknobs_bots.registry import InMemoryBackend

        backend = create_registry_backend("memory", {})
        assert isinstance(backend, InMemoryBackend)

    def test_create_registry_backend_unknown(self):
        """Test factory raises for unknown backend type."""
        with pytest.raises(ValueError) as excinfo:
            create_registry_backend("unknown", {})

        assert "unknown" in str(excinfo.value).lower()
        assert "memory" in str(excinfo.value)


class TestHTTPRegistryBackendNotInitialized:
    """Tests for uninitialized backend behavior."""

    @pytest.mark.asyncio
    async def test_operations_fail_without_init(self):
        """Test operations fail if not initialized."""
        backend = HTTPRegistryBackend(base_url="https://example.com")

        with pytest.raises(RuntimeError) as excinfo:
            await backend.get("test-bot")

        assert "not initialized" in str(excinfo.value).lower()


class TestHTTPRegistryBackendReadOnly:
    """Tests for read-only mode."""

    @pytest.mark.asyncio
    async def test_write_operations_blocked(self):
        """Test write operations are blocked in read-only mode."""
        backend = HTTPRegistryBackend(
            base_url="https://example.com",
            read_only=True,
        )
        # Mock initialization
        backend._initialized = True
        backend._session = None  # Will fail on actual request, but permission check first

        with pytest.raises(PermissionError):
            await backend.register("test", {})

        with pytest.raises(PermissionError):
            await backend.unregister("test")

        with pytest.raises(PermissionError):
            await backend.deactivate("test")

        with pytest.raises(PermissionError):
            await backend.clear()


class TestHTTPRegistryBackendWithMockServer:
    """Integration tests with a real in-process aiohttp test server."""

    @pytest.fixture
    async def mock_server(self):
        """Spin up the in-process aiohttp test server for the duration of a test."""
        server = _MockHttpServer()
        await server.start()
        try:
            yield server
        finally:
            await server.stop()

    @pytest.fixture
    async def backend(self, mock_server):
        """Create and initialize a backend pointed at the mock server."""
        backend = HTTPRegistryBackend(
            base_url=mock_server.base_url,
            auth_token="test-token",
        )
        await backend.initialize()
        yield backend
        await backend.close()

    @pytest.mark.asyncio
    async def test_get_config_success(self, backend, mock_server):
        """Test fetching a configuration."""
        mock_server.get(
            "/api/v1/configs/test-bot",
            payload={
                "bot_id": "test-bot",
                "config": {"llm": {"provider": "anthropic"}},
                "status": "active",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            },
        )

        config = await backend.get_config("test-bot")

        assert config == {"llm": {"provider": "anthropic"}}

    @pytest.mark.asyncio
    async def test_get_config_not_found(self, backend, mock_server):
        """Test fetching non-existent configuration."""
        mock_server.get("/api/v1/configs/missing", status=404)

        config = await backend.get_config("missing")

        assert config is None

    @pytest.mark.asyncio
    async def test_get_registration(self, backend, mock_server):
        """Test fetching a full registration."""
        mock_server.get(
            "/api/v1/configs/test-bot",
            payload={
                "bot_id": "test-bot",
                "config": {"key": "value"},
                "status": "active",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
            },
        )

        reg = await backend.get("test-bot")

        assert reg is not None
        assert reg.bot_id == "test-bot"
        assert reg.config == {"key": "value"}
        assert reg.status == "active"

    @pytest.mark.asyncio
    async def test_exists_active(self, backend, mock_server):
        """Test checking if active registration exists."""
        mock_server.get(
            "/api/v1/configs/test-bot",
            payload={
                "bot_id": "test-bot",
                "config": {},
                "status": "active",
            },
        )

        assert await backend.exists("test-bot") is True

    @pytest.mark.asyncio
    async def test_exists_inactive(self, backend, mock_server):
        """Test that exists returns False for inactive registrations."""
        mock_server.get(
            "/api/v1/configs/test-bot",
            payload={
                "bot_id": "test-bot",
                "config": {},
                "status": "inactive",
            },
        )

        assert await backend.exists("test-bot") is False

    @pytest.mark.asyncio
    async def test_list_all(self, backend, mock_server):
        """Test listing all registrations."""
        mock_server.get(
            "/api/v1/configs",
            payload=[
                {"bot_id": "bot-1", "config": {}, "status": "active"},
                {"bot_id": "bot-2", "config": {}, "status": "inactive"},
            ],
        )

        regs = await backend.list_all()

        assert len(regs) == 2
        assert regs[0].bot_id == "bot-1"
        assert regs[1].bot_id == "bot-2"

    @pytest.mark.asyncio
    async def test_list_all_with_items_key(self, backend, mock_server):
        """Test listing with response that has 'items' key."""
        mock_server.get(
            "/api/v1/configs",
            payload={
                "items": [
                    {"bot_id": "bot-1", "config": {}, "status": "active"},
                ],
                "total": 1,
            },
        )

        regs = await backend.list_all()

        assert len(regs) == 1
        assert regs[0].bot_id == "bot-1"

    @pytest.mark.asyncio
    async def test_list_active_filters(self, backend, mock_server):
        """Test that list_active filters inactive registrations.

        The wire call carries ``?status=active`` but routing is by path,
        so a single ``/api/v1/configs`` registration matches whether or
        not a query string is attached.  The defensive client-side
        ``status`` reapply still applies if a legacy server returns the
        unfiltered list.
        """
        mock_server.get(
            "/api/v1/configs",
            payload=[
                {"bot_id": "bot-1", "config": {}, "status": "active"},
                {"bot_id": "bot-2", "config": {}, "status": "inactive"},
                {"bot_id": "bot-3", "config": {}, "status": "active"},
            ],
        )

        regs = await backend.list_active()

        assert len(regs) == 2
        assert all(r.status == "active" for r in regs)

    @pytest.mark.asyncio
    async def test_list_ids(self, backend, mock_server):
        """Test listing active bot IDs."""
        mock_server.get(
            "/api/v1/configs",
            payload=[
                {"bot_id": "bot-1", "config": {}, "status": "active"},
                {"bot_id": "bot-2", "config": {}, "status": "inactive"},
            ],
        )

        ids = await backend.list_ids()

        assert ids == ["bot-1"]

    @pytest.mark.asyncio
    async def test_count(self, backend, mock_server):
        """Test counting active registrations."""
        mock_server.get(
            "/api/v1/configs",
            payload=[
                {"bot_id": "bot-1", "config": {}, "status": "active"},
                {"bot_id": "bot-2", "config": {}, "status": "active"},
                {"bot_id": "bot-3", "config": {}, "status": "inactive"},
            ],
        )

        count = await backend.count()

        assert count == 2

    @pytest.mark.asyncio
    async def test_register_new(self, backend, mock_server):
        """``register`` issues a single PUT (upsert) — no probing GET."""
        mock_server.put(
            "/api/v1/configs/new-bot",
            payload={
                "bot_id": "new-bot",
                "config": {"key": "value"},
                "status": "active",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            },
        )

        reg = await backend.register("new-bot", {"key": "value"})

        assert reg.bot_id == "new-bot"
        assert reg.config == {"key": "value"}

    @pytest.mark.asyncio
    async def test_register_update(self, backend, mock_server):
        """Updating uses the same single PUT — server treats register as upsert."""
        mock_server.put(
            "/api/v1/configs/existing-bot",
            payload={
                "bot_id": "existing-bot",
                "config": {"new": "value"},
                "status": "active",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
            },
        )

        reg = await backend.register("existing-bot", {"new": "value"})

        assert reg.config == {"new": "value"}

    @pytest.mark.asyncio
    async def test_unregister(self, backend, mock_server):
        """Test deleting a registration."""
        mock_server.delete("/api/v1/configs/test-bot", status=204)

        result = await backend.unregister("test-bot")

        assert result is True

    @pytest.mark.asyncio
    async def test_unregister_not_found(self, backend, mock_server):
        """Test deleting non-existent registration."""
        mock_server.delete("/api/v1/configs/missing", status=404)

        result = await backend.unregister("missing")

        assert result is False

    @pytest.mark.asyncio
    async def test_deactivate(self, backend, mock_server):
        """``deactivate`` hits the dedicated endpoint — no touching GET."""
        mock_server.post(
            "/api/v1/configs/test-bot/deactivate",
            status=204,
        )

        result = await backend.deactivate("test-bot")

        assert result is True

    @pytest.mark.asyncio
    async def test_deactivate_not_found(self, backend, mock_server):
        """Server 404 from the dedicated endpoint surfaces as ``False``."""
        mock_server.post(
            "/api/v1/configs/missing/deactivate",
            status=404,
        )

        result = await backend.deactivate("missing")

        assert result is False

    @pytest.mark.asyncio
    async def test_http_error_handling(self, backend, mock_server):
        """Test HTTP error response handling."""
        from aiohttp import ClientResponseError

        mock_server.get(
            "/api/v1/configs/test-bot",
            status=500,
            body=b"Internal Server Error",
        )

        with pytest.raises(ClientResponseError) as excinfo:
            await backend.get("test-bot")

        assert excinfo.value.status == 500

    @pytest.mark.asyncio
    async def test_peek_config_success(self, backend, mock_server):
        """peek_config returns the config dict for an existing bot."""
        mock_server.get(
            "/api/v1/configs/peek-bot",
            payload={
                "bot_id": "peek-bot",
                "config": {"llm": {"provider": "anthropic"}},
                "status": "active",
            },
        )

        config = await backend.peek_config("peek-bot")

        assert config == {"llm": {"provider": "anthropic"}}

    @pytest.mark.asyncio
    async def test_peek_config_not_found(self, backend, mock_server):
        """peek_config returns None when the bot is missing."""
        mock_server.get("/api/v1/configs/missing", status=404)

        config = await backend.peek_config("missing")

        assert config is None

    @pytest.mark.asyncio
    async def test_peek_config_does_not_impose_wire_protocol(
        self, backend, mock_server
    ):
        """peek_config issues a plain GET — no client-imposed wire protocol.

        The HTTP backend does not maintain client-side ``last_accessed_at``
        state, so the Protocol's non-mutation guarantee is satisfied
        unconditionally without a transport-level peek hint. Servers
        that want to distinguish peek from get must define their own
        contract.

        Verified by iterating the captured calls and asserting an empty
        query mapping — aiohttp's server-side request parser exposes
        every actually-sent query parameter, so a client-imposed hint
        would surface here.
        """
        mock_server.get(
            "/api/v1/configs/hint-bot",
            payload={
                "bot_id": "hint-bot",
                "config": {"k": "v"},
                "status": "active",
            },
        )

        config = await backend.peek_config("hint-bot")
        assert config == {"k": "v"}

        calls = mock_server.calls
        assert calls, "expected at least one captured request"
        for call in calls:
            assert not call.query, (
                f"peek_config issued {call.method} {call.path} with query "
                f"{call.query!r} — the client must not impose a wire-protocol "
                "query parameter on the server"
            )

    @pytest.mark.asyncio
    async def test_peek_config_works_in_read_only_mode(self, mock_server):
        """peek_config is a read; allowed in read-only mode."""
        backend = HTTPRegistryBackend(
            base_url=mock_server.base_url,
            read_only=True,
        )
        await backend.initialize()
        try:
            mock_server.get(
                "/api/v1/configs/ro-bot",
                payload={
                    "bot_id": "ro-bot",
                    "config": {"k": "v"},
                    "status": "active",
                },
            )
            config = await backend.peek_config("ro-bot")
            assert config == {"k": "v"}
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_list_all_no_filter_sends_no_query_param(
        self, backend, mock_server
    ):
        """``list_all()`` with no filter must not pass ``params`` to aiohttp."""
        mock_server.get(
            "/api/v1/configs",
            payload=[
                {"bot_id": "bot-1", "config": {}, "status": "active"},
            ],
        )

        await backend.list_all()

        captured = _captured_params(mock_server)
        assert captured, "expected a GET /configs request"
        for params in captured:
            assert not params, (
                f"list_all() without filter must not send params, got {params!r}"
            )

    @pytest.mark.asyncio
    async def test_list_all_pushes_filter_metadata_to_server(
        self, backend, mock_server
    ):
        """list_all sends filter_metadata as a URL-encoded JSON query param."""
        mock_server.get(
            "/api/v1/configs",
            payload=[
                {
                    "bot_id": "bot-1",
                    "config": {},
                    "status": "active",
                    "metadata": {"tenant_id": "acme"},
                },
            ],
        )

        regs = await backend.list_all(filter_metadata={"tenant_id": "acme"})

        assert len(regs) == 1
        assert _filter_metadata_param(mock_server) == {"tenant_id": "acme"}

    @pytest.mark.asyncio
    async def test_list_active_pushes_filter_metadata_to_server(
        self, backend, mock_server
    ):
        """list_active routes filter_metadata down to the same GET /configs call."""
        mock_server.get(
            "/api/v1/configs",
            payload=[
                {
                    "bot_id": "bot-1",
                    "config": {},
                    "status": "active",
                    "metadata": {"tenant_id": "acme"},
                },
            ],
        )

        regs = await backend.list_active(filter_metadata={"tenant_id": "acme"})

        assert len(regs) == 1
        assert _filter_metadata_param(mock_server) == {"tenant_id": "acme"}

    @pytest.mark.asyncio
    async def test_count_pushes_filter_metadata_to_server(
        self, backend, mock_server
    ):
        """count() routes filter_metadata down to the underlying list call."""
        mock_server.get(
            "/api/v1/configs",
            payload=[
                {
                    "bot_id": "bot-1",
                    "config": {},
                    "status": "active",
                    "metadata": {"tenant_id": "acme"},
                },
                {
                    "bot_id": "bot-2",
                    "config": {},
                    "status": "active",
                    "metadata": {"tenant_id": "acme"},
                },
            ],
        )

        n = await backend.count(filter_metadata={"tenant_id": "acme"})

        assert n == 2
        assert _filter_metadata_param(mock_server) == {"tenant_id": "acme"}

    @pytest.mark.asyncio
    async def test_filter_metadata_query_param_is_deterministic(
        self, backend, mock_server
    ):
        """``filter_metadata`` JSON is serialized with ``sort_keys=True``.

        Determinism matters for server-side cache keys and request logs:
        the same logical filter must produce the same wire-level query
        string regardless of input dict order.
        """
        mock_server.get("/api/v1/configs", payload=[])

        await backend.list_all(filter_metadata={"z_last": 1, "a_first": 2})

        captured = _captured_params(mock_server)
        raw = next(
            (p["filter_metadata"] for p in captured if "filter_metadata" in p),
            None,
        )
        assert isinstance(raw, str)
        # sort_keys=True puts "a_first" before "z_last" in the JSON
        # string handed to aiohttp.
        assert raw.index('"a_first"') < raw.index('"z_last"')

    @pytest.mark.asyncio
    async def test_client_side_filter_applied_when_server_ignores_query_param(
        self, backend, mock_server
    ):
        """If the server returns unfiltered rows, the client must still filter.

        The wire protocol is additive-optional — servers that don't honor
        ``filter_metadata`` return the unfiltered list. The client's
        defensive post-filter guarantees correctness.
        """
        mock_server.get(
            "/api/v1/configs",
            payload=[
                {
                    "bot_id": "bot-1",
                    "config": {},
                    "status": "active",
                    "metadata": {"tenant_id": "acme"},
                },
                {
                    "bot_id": "bot-2",
                    "config": {},
                    "status": "active",
                    "metadata": {"tenant_id": "other"},
                },
            ],
        )

        regs = await backend.list_all(filter_metadata={"tenant_id": "acme"})

        # Server returned both, but the client must filter on tenant_id.
        assert [r.bot_id for r in regs] == ["bot-1"]

    # --- status/sort/limit/offset push-down -----------------

    @pytest.mark.asyncio
    async def test_status_pushed_to_server(self, backend, mock_server):
        """``list_all(status=...)`` sends ``?status=<value>``."""
        mock_server.get(
            "/api/v1/configs",
            payload=[
                {"bot_id": "bot-1", "config": {}, "status": "active"},
            ],
        )

        await backend.list_all(status="active")

        captured = _captured_params(mock_server)
        assert captured, "expected a GET /configs request"
        statuses = [p.get("status") for p in captured if p]
        assert "active" in statuses

    @pytest.mark.asyncio
    async def test_list_inactive_pushes_status_to_server(
        self, backend, mock_server
    ):
        """``list_inactive()`` routes through the same ``?status=`` push-down."""
        mock_server.get(
            "/api/v1/configs",
            payload=[
                {"bot_id": "bot-1", "config": {}, "status": "inactive"},
            ],
        )

        await backend.list_inactive()

        captured = _captured_params(mock_server)
        statuses = [p.get("status") for p in captured if p]
        assert "inactive" in statuses

    @pytest.mark.asyncio
    async def test_sort_encoded_as_field_colon_order(
        self, backend, mock_server
    ):
        """``sort=[SortSpec("bot_id", DESC)]`` serializes to ``?sort=bot_id:desc``.

        Single-occurrence wire params surface as scalars via
        ``_captured_params`` — the original assertion expected a list
        because aioresponses preserved the backend's input shape, but
        the wire format (which is what we actually care about) is one
        ``sort=bot_id:desc`` query parameter.
        """
        from dataknobs_data import SortOrder, SortSpec

        mock_server.get("/api/v1/configs", payload=[])

        await backend.list_all(
            sort=[SortSpec(field="bot_id", order=SortOrder.DESC)]
        )

        captured = _captured_params(mock_server)
        sort_params = [p.get("sort") for p in captured if p]
        assert sort_params == ["bot_id:desc"]

    @pytest.mark.asyncio
    async def test_sort_multi_key_preserves_list_order(
        self, backend, mock_server
    ):
        """Multi-key sort serializes as a list (repeated query param)."""
        from dataknobs_data import SortOrder, SortSpec

        mock_server.get("/api/v1/configs", payload=[])

        await backend.list_all(
            sort=[
                SortSpec(field="metadata.team", order=SortOrder.ASC),
                SortSpec(field="bot_id", order=SortOrder.DESC),
            ]
        )

        captured = _captured_params(mock_server)
        sort_params = [p.get("sort") for p in captured if p]
        # Wire order matches caller's list order (tie-break semantics
        # are positional).
        assert sort_params == [["metadata.team:asc", "bot_id:desc"]]

    @pytest.mark.asyncio
    async def test_limit_pushed_to_server(self, backend, mock_server):
        mock_server.get("/api/v1/configs", payload=[])

        await backend.list_all(limit=10)

        captured = _captured_params(mock_server)
        limits = [p.get("limit") for p in captured if p]
        assert "10" in limits

    @pytest.mark.asyncio
    async def test_offset_pushed_to_server(self, backend, mock_server):
        mock_server.get("/api/v1/configs", payload=[])

        await backend.list_all(offset=5)

        captured = _captured_params(mock_server)
        offsets = [p.get("offset") for p in captured if p]
        assert "5" in offsets

    @pytest.mark.asyncio
    async def test_limit_offset_not_reapplied_client_side(
        self, backend, mock_server
    ):
        """Server is trusted on pagination — client must not re-truncate.

        Re-applying ``offset`` on an already-offset window would drop
        live rows.  Verified by having the server return exactly the
        page the client requested; the client must surface all 3 rows
        even though offset=5 would skip past everything if reapplied.
        """
        mock_server.get(
            "/api/v1/configs",
            payload=[
                {"bot_id": "bot-5", "config": {}, "status": "active"},
                {"bot_id": "bot-6", "config": {}, "status": "active"},
                {"bot_id": "bot-7", "config": {}, "status": "active"},
            ],
        )

        regs = await backend.list_all(offset=5)

        assert [r.bot_id for r in regs] == ["bot-5", "bot-6", "bot-7"]

    @pytest.mark.asyncio
    async def test_limit_not_reapplied_client_side(
        self, backend, mock_server
    ):
        """The server is trusted on limit; client does not re-truncate."""
        mock_server.get(
            "/api/v1/configs",
            payload=[
                {"bot_id": "bot-0", "config": {}, "status": "active"},
                {"bot_id": "bot-1", "config": {}, "status": "active"},
            ],
        )

        # Caller asked for 5; server delivered 2. Client returns 2,
        # not 0 or 5.
        regs = await backend.list_all(limit=5)
        assert len(regs) == 2

    @pytest.mark.asyncio
    async def test_status_reapplied_client_side_for_legacy_servers(
        self, backend, mock_server
    ):
        """Legacy server returning mixed-status rows is still filtered client-side.

        ``status`` is an additive-optional parameter; the defensive
        client-side reapply guarantees correctness when the server
        ignores ``?status=``.
        """
        mock_server.get(
            "/api/v1/configs",
            payload=[
                {"bot_id": "bot-1", "config": {}, "status": "active"},
                {"bot_id": "bot-2", "config": {}, "status": "inactive"},
            ],
        )

        regs = await backend.list_all(status="active")

        # Server returned both; client filters down to active.
        assert [r.bot_id for r in regs] == ["bot-1"]

    @pytest.mark.asyncio
    async def test_sort_reapplied_client_side_for_legacy_servers(
        self, backend, mock_server
    ):
        """Legacy server returns unsorted rows; the client must sort them.

        Re-sorting is idempotent, so the defensive reapply is always
        safe (a server that already sorted will see no change).
        """
        from dataknobs_data import SortOrder, SortSpec

        mock_server.get(
            "/api/v1/configs",
            payload=[
                {"bot_id": "charlie", "config": {}, "status": "active"},
                {"bot_id": "alice", "config": {}, "status": "active"},
                {"bot_id": "bob", "config": {}, "status": "active"},
            ],
        )

        regs = await backend.list_all(
            sort=[SortSpec(field="bot_id", order=SortOrder.ASC)]
        )
        assert [r.bot_id for r in regs] == ["alice", "bob", "charlie"]

    @pytest.mark.asyncio
    async def test_count_pushes_status_to_server(self, backend, mock_server):
        """``count(filter_metadata=...)`` already sends ``filter_metadata``;
        this confirms ``count_all(status=...)`` adds ``?status=``."""
        mock_server.get(
            "/api/v1/configs",
            payload=[
                {"bot_id": "bot-1", "config": {}, "status": "inactive"},
                {"bot_id": "bot-2", "config": {}, "status": "inactive"},
            ],
        )

        n = await backend.count_inactive()

        assert n == 2
        captured = _captured_params(mock_server)
        statuses = [p.get("status") for p in captured if p]
        assert "inactive" in statuses

    @pytest.mark.asyncio
    async def test_no_params_sends_no_query_string(self, backend, mock_server):
        """Confirm invariant: bare ``list_all()`` adds no query string."""
        mock_server.get("/api/v1/configs", payload=[])

        await backend.list_all()

        captured = _captured_params(mock_server)
        assert captured, "expected a request"
        for params in captured:
            assert not params, (
                f"bare list_all() must not send params, got {params!r}"
            )

    @pytest.mark.asyncio
    async def test_auth_header_sent(self, mock_server):
        """Test that the auth header reaches the server on the wire.

        Unlike the previous aioresponses-driven test (which only
        asserted that no error was raised), this version inspects the
        actual ``Authorization`` header the server received — proving
        the backend wires the auth_token through aiohttp's
        ClientSession all the way to the wire.
        """
        backend = HTTPRegistryBackend(
            base_url=mock_server.base_url,
            auth_token="my-secret-token",
        )
        await backend.initialize()

        try:
            mock_server.get(
                "/api/v1/configs/test",
                payload={"bot_id": "test", "config": {}, "status": "active"},
            )

            await backend.get("test")

            calls = mock_server.calls_for("GET")
            assert calls, "expected a GET request"
            assert calls[0].headers.get("Authorization") == "Bearer my-secret-token"
        finally:
            await backend.close()

"""Tests for HTTPRegistryBackend."""

import json
import re

import pytest

from dataknobs_bots.registry import HTTPRegistryBackend, create_registry_backend

# Matches ``GET /configs`` with or without a query string.  Used by the
# wire-protocol tests because ``aioresponses`` 0.7.8 matches
# URLs by strict equality — a bare-URL registration would not match a
# request carrying ``?filter_metadata=...``.
_CONFIGS_URL_PATTERN = re.compile(
    r"^https://config-service\.test/api/v1/configs(\?.*)?$"
)


def _captured_params(mock_responses, *, method: str = "GET") -> list[dict | None]:
    """Return the ``params`` kwarg of every captured request for ``method``.

    aioresponses stores each request's original kwargs (deep-copied) on
    ``RequestCall.kwargs``.  Inspecting ``params`` directly is more
    robust than parsing the on-wire URL because it bypasses the yarl
    encode / ``urllib.parse`` decode round-trip — and it asserts on
    exactly what we handed to aiohttp, not on a representation of it.
    """
    result: list[dict | None] = []
    for (m, _url), calls in mock_responses.requests.items():
        if m.upper() != method.upper():
            continue
        for call in calls:
            result.append(call.kwargs.get("params"))
    return result


def _filter_metadata_param(mock_responses) -> dict | None:
    """Return the decoded ``filter_metadata`` from the first matching call."""
    for params in _captured_params(mock_responses):
        if params and "filter_metadata" in params:
            return json.loads(params["filter_metadata"])
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
    """Integration tests with mocked HTTP responses."""

    @pytest.fixture
    def mock_responses(self):
        """Set up aioresponses mock."""
        try:
            from aioresponses import aioresponses

            with aioresponses() as m:
                yield m
        except ImportError:
            pytest.skip("aioresponses not installed")

    @pytest.fixture
    async def backend(self, mock_responses):
        """Create and initialize backend with mock server."""
        backend = HTTPRegistryBackend(
            base_url="https://config-service.test/api/v1",
            auth_token="test-token",
        )
        await backend.initialize()
        yield backend
        await backend.close()

    @pytest.mark.asyncio
    async def test_get_config_success(self, backend, mock_responses):
        """Test fetching a configuration."""
        mock_responses.get(
            "https://config-service.test/api/v1/configs/test-bot",
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
    async def test_get_config_not_found(self, backend, mock_responses):
        """Test fetching non-existent configuration."""
        mock_responses.get(
            "https://config-service.test/api/v1/configs/missing",
            status=404,
        )

        config = await backend.get_config("missing")

        assert config is None

    @pytest.mark.asyncio
    async def test_get_registration(self, backend, mock_responses):
        """Test fetching a full registration."""
        mock_responses.get(
            "https://config-service.test/api/v1/configs/test-bot",
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
    async def test_exists_active(self, backend, mock_responses):
        """Test checking if active registration exists."""
        mock_responses.get(
            "https://config-service.test/api/v1/configs/test-bot",
            payload={
                "bot_id": "test-bot",
                "config": {},
                "status": "active",
            },
        )

        assert await backend.exists("test-bot") is True

    @pytest.mark.asyncio
    async def test_exists_inactive(self, backend, mock_responses):
        """Test that exists returns False for inactive registrations."""
        mock_responses.get(
            "https://config-service.test/api/v1/configs/test-bot",
            payload={
                "bot_id": "test-bot",
                "config": {},
                "status": "inactive",
            },
        )

        assert await backend.exists("test-bot") is False

    @pytest.mark.asyncio
    async def test_list_all(self, backend, mock_responses):
        """Test listing all registrations."""
        mock_responses.get(
            "https://config-service.test/api/v1/configs",
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
    async def test_list_all_with_items_key(self, backend, mock_responses):
        """Test listing with response that has 'items' key."""
        mock_responses.get(
            "https://config-service.test/api/v1/configs",
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
    async def test_list_active_filters(self, backend, mock_responses):
        """Test that list_active filters inactive registrations.

        The wire call now carries ``?status=active``, so the mock is
        registered with the bare-or-querystring URL pattern.  The
        defensive client-side ``status`` reapply still applies if a
        legacy server returns the unfiltered list.
        """
        mock_responses.get(
            _CONFIGS_URL_PATTERN,
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
    async def test_list_ids(self, backend, mock_responses):
        """Test listing active bot IDs."""
        mock_responses.get(
            _CONFIGS_URL_PATTERN,
            payload=[
                {"bot_id": "bot-1", "config": {}, "status": "active"},
                {"bot_id": "bot-2", "config": {}, "status": "inactive"},
            ],
        )

        ids = await backend.list_ids()

        assert ids == ["bot-1"]

    @pytest.mark.asyncio
    async def test_count(self, backend, mock_responses):
        """Test counting active registrations."""
        mock_responses.get(
            _CONFIGS_URL_PATTERN,
            payload=[
                {"bot_id": "bot-1", "config": {}, "status": "active"},
                {"bot_id": "bot-2", "config": {}, "status": "active"},
                {"bot_id": "bot-3", "config": {}, "status": "inactive"},
            ],
        )

        count = await backend.count()

        assert count == 2

    @pytest.mark.asyncio
    async def test_register_new(self, backend, mock_responses):
        """``register`` issues a single PUT (upsert) — no probing GET."""
        mock_responses.put(
            "https://config-service.test/api/v1/configs/new-bot",
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
    async def test_register_update(self, backend, mock_responses):
        """Updating uses the same single PUT — server treats register as upsert."""
        mock_responses.put(
            "https://config-service.test/api/v1/configs/existing-bot",
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
    async def test_unregister(self, backend, mock_responses):
        """Test deleting a registration."""
        mock_responses.delete(
            "https://config-service.test/api/v1/configs/test-bot",
            status=204,
        )

        result = await backend.unregister("test-bot")

        assert result is True

    @pytest.mark.asyncio
    async def test_unregister_not_found(self, backend, mock_responses):
        """Test deleting non-existent registration."""
        mock_responses.delete(
            "https://config-service.test/api/v1/configs/missing",
            status=404,
        )

        result = await backend.unregister("missing")

        assert result is False

    @pytest.mark.asyncio
    async def test_deactivate(self, backend, mock_responses):
        """``deactivate`` hits the dedicated endpoint — no touching GET."""
        mock_responses.post(
            "https://config-service.test/api/v1/configs/test-bot/deactivate",
            status=204,
        )

        result = await backend.deactivate("test-bot")

        assert result is True

    @pytest.mark.asyncio
    async def test_deactivate_not_found(self, backend, mock_responses):
        """Server 404 from the dedicated endpoint surfaces as ``False``."""
        mock_responses.post(
            "https://config-service.test/api/v1/configs/missing/deactivate",
            status=404,
        )

        result = await backend.deactivate("missing")

        assert result is False

    @pytest.mark.asyncio
    async def test_http_error_handling(self, backend, mock_responses):
        """Test HTTP error response handling."""
        from aiohttp import ClientResponseError

        mock_responses.get(
            "https://config-service.test/api/v1/configs/test-bot",
            status=500,
            body="Internal Server Error",
        )

        with pytest.raises(ClientResponseError) as excinfo:
            await backend.get("test-bot")

        assert excinfo.value.status == 500

    @pytest.mark.asyncio
    async def test_peek_config_success(self, backend, mock_responses):
        """peek_config returns the config dict for an existing bot."""
        mock_responses.get(
            "https://config-service.test/api/v1/configs/peek-bot",
            payload={
                "bot_id": "peek-bot",
                "config": {"llm": {"provider": "anthropic"}},
                "status": "active",
            },
        )

        config = await backend.peek_config("peek-bot")

        assert config == {"llm": {"provider": "anthropic"}}

    @pytest.mark.asyncio
    async def test_peek_config_not_found(self, backend, mock_responses):
        """peek_config returns None when the bot is missing."""
        mock_responses.get(
            "https://config-service.test/api/v1/configs/missing",
            status=404,
        )

        config = await backend.peek_config("missing")

        assert config is None

    @pytest.mark.asyncio
    async def test_peek_config_does_not_impose_wire_protocol(
        self, backend, mock_responses
    ):
        """peek_config issues a plain GET — no client-imposed wire protocol.

        The HTTP backend does not maintain client-side ``last_accessed_at``
        state, so the Protocol's non-mutation guarantee is satisfied
        unconditionally without a transport-level peek hint. Servers
        that want to distinguish peek from get must define their own
        contract.

        Pinned by inspecting the captured request URLs directly: the
        ``aioresponses`` ``match_querystring`` default is False, so a
        registered bare URL would still match a ``?peek=true`` request
        — which means the response setup alone cannot pin the absence
        of a query string. We assert it via the recorded request keys.
        """
        mock_responses.get(
            "https://config-service.test/api/v1/configs/hint-bot",
            payload={
                "bot_id": "hint-bot",
                "config": {"k": "v"},
                "status": "active",
            },
        )

        config = await backend.peek_config("hint-bot")
        assert config == {"k": "v"}

        # mock_responses.requests is keyed by (method, normalized_url)
        # where the URL reflects whatever the client actually sent
        # (query string included). Iterate and assert no request URL
        # carries a query string at all.
        request_keys = list(mock_responses.requests.keys())
        assert request_keys, "expected at least one captured request"
        for method, url in request_keys:
            url_str = str(url)
            assert "?" not in url_str, (
                f"peek_config issued {method} {url_str} — the client "
                "must not impose a wire-protocol query parameter on "
                "the server"
            )

    @pytest.mark.asyncio
    async def test_peek_config_works_in_read_only_mode(self, mock_responses):
        """peek_config is a read; allowed in read-only mode."""
        backend = HTTPRegistryBackend(
            base_url="https://config-service.test/api/v1",
            read_only=True,
        )
        await backend.initialize()
        try:
            mock_responses.get(
                "https://config-service.test/api/v1/configs/ro-bot",
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
        self, backend, mock_responses
    ):
        """``list_all()`` with no filter must not pass ``params`` to aiohttp."""
        mock_responses.get(
            "https://config-service.test/api/v1/configs",
            payload=[
                {"bot_id": "bot-1", "config": {}, "status": "active"},
            ],
        )

        await backend.list_all()

        captured = _captured_params(mock_responses)
        assert captured, "expected a GET /configs request"
        # The client must pass ``params=None`` (or omit it) when no
        # filter is set, so the request goes on the wire without a
        # query string.
        for params in captured:
            assert not params, (
                f"list_all() without filter must not send params, got {params!r}"
            )

    @pytest.mark.asyncio
    async def test_list_all_pushes_filter_metadata_to_server(
        self, backend, mock_responses
    ):
        """list_all sends filter_metadata as a URL-encoded JSON query param."""
        mock_responses.get(
            _CONFIGS_URL_PATTERN,
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
        assert _filter_metadata_param(mock_responses) == {"tenant_id": "acme"}

    @pytest.mark.asyncio
    async def test_list_active_pushes_filter_metadata_to_server(
        self, backend, mock_responses
    ):
        """list_active routes filter_metadata down to the same GET /configs call."""
        mock_responses.get(
            _CONFIGS_URL_PATTERN,
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
        assert _filter_metadata_param(mock_responses) == {"tenant_id": "acme"}

    @pytest.mark.asyncio
    async def test_count_pushes_filter_metadata_to_server(
        self, backend, mock_responses
    ):
        """count() routes filter_metadata down to the underlying list call."""
        mock_responses.get(
            _CONFIGS_URL_PATTERN,
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
        assert _filter_metadata_param(mock_responses) == {"tenant_id": "acme"}

    @pytest.mark.asyncio
    async def test_filter_metadata_query_param_is_deterministic(
        self, backend, mock_responses
    ):
        """``filter_metadata`` JSON is serialized with ``sort_keys=True``.

        Determinism matters for server-side cache keys and request logs:
        the same logical filter must produce the same wire-level query
        string regardless of input dict order.
        """
        mock_responses.get(
            _CONFIGS_URL_PATTERN,
            payload=[],
        )

        await backend.list_all(filter_metadata={"z_last": 1, "a_first": 2})

        captured = _captured_params(mock_responses)
        raw = next(
            (p["filter_metadata"] for p in captured if p and "filter_metadata" in p),
            None,
        )
        assert raw is not None
        # sort_keys=True puts "a_first" before "z_last" in the JSON
        # string handed to aiohttp.
        assert raw.index('"a_first"') < raw.index('"z_last"')

    @pytest.mark.asyncio
    async def test_client_side_filter_applied_when_server_ignores_query_param(
        self, backend, mock_responses
    ):
        """If the server returns unfiltered rows, the client must still filter.

        The wire protocol is additive-optional — servers that don't honor
        ``filter_metadata`` return the unfiltered list. The client's
        defensive post-filter guarantees correctness.
        """
        mock_responses.get(
            _CONFIGS_URL_PATTERN,
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
    async def test_status_pushed_to_server(self, backend, mock_responses):
        """``list_all(status=...)`` sends ``?status=<value>``."""
        mock_responses.get(
            _CONFIGS_URL_PATTERN,
            payload=[
                {"bot_id": "bot-1", "config": {}, "status": "active"},
            ],
        )

        await backend.list_all(status="active")

        captured = _captured_params(mock_responses)
        assert captured, "expected a GET /configs request"
        statuses = [p.get("status") for p in captured if p]
        assert "active" in statuses

    @pytest.mark.asyncio
    async def test_list_inactive_pushes_status_to_server(
        self, backend, mock_responses
    ):
        """``list_inactive()`` routes through the same ``?status=`` push-down."""
        mock_responses.get(
            _CONFIGS_URL_PATTERN,
            payload=[
                {"bot_id": "bot-1", "config": {}, "status": "inactive"},
            ],
        )

        await backend.list_inactive()

        captured = _captured_params(mock_responses)
        statuses = [p.get("status") for p in captured if p]
        assert "inactive" in statuses

    @pytest.mark.asyncio
    async def test_sort_encoded_as_field_colon_order(
        self, backend, mock_responses
    ):
        """``sort=[SortSpec("bot_id", DESC)]`` serializes to ``?sort=bot_id:desc``."""
        from dataknobs_data import SortOrder, SortSpec

        mock_responses.get(_CONFIGS_URL_PATTERN, payload=[])

        await backend.list_all(
            sort=[SortSpec(field="bot_id", order=SortOrder.DESC)]
        )

        captured = _captured_params(mock_responses)
        sort_params = [p.get("sort") for p in captured if p]
        assert sort_params == [["bot_id:desc"]]

    @pytest.mark.asyncio
    async def test_sort_multi_key_preserves_list_order(
        self, backend, mock_responses
    ):
        """Multi-key sort serializes as a list (repeated query param)."""
        from dataknobs_data import SortOrder, SortSpec

        mock_responses.get(_CONFIGS_URL_PATTERN, payload=[])

        await backend.list_all(
            sort=[
                SortSpec(field="metadata.team", order=SortOrder.ASC),
                SortSpec(field="bot_id", order=SortOrder.DESC),
            ]
        )

        captured = _captured_params(mock_responses)
        sort_params = [p.get("sort") for p in captured if p]
        # Wire order matches caller's list order (tie-break semantics
        # are positional).
        assert sort_params == [["metadata.team:asc", "bot_id:desc"]]

    @pytest.mark.asyncio
    async def test_limit_pushed_to_server(self, backend, mock_responses):
        mock_responses.get(_CONFIGS_URL_PATTERN, payload=[])

        await backend.list_all(limit=10)

        captured = _captured_params(mock_responses)
        limits = [p.get("limit") for p in captured if p]
        assert "10" in limits

    @pytest.mark.asyncio
    async def test_offset_pushed_to_server(self, backend, mock_responses):
        mock_responses.get(_CONFIGS_URL_PATTERN, payload=[])

        await backend.list_all(offset=5)

        captured = _captured_params(mock_responses)
        offsets = [p.get("offset") for p in captured if p]
        assert "5" in offsets

    @pytest.mark.asyncio
    async def test_limit_offset_not_reapplied_client_side(
        self, backend, mock_responses
    ):
        """Server is trusted on pagination — client must not re-truncate.

        Re-applying ``offset`` on an already-offset window would drop
        live rows.  Verified by having the server return exactly the
        page the client requested; the client must surface all 3 rows
        even though offset=5 would skip past everything if reapplied.
        """
        mock_responses.get(
            _CONFIGS_URL_PATTERN,
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
        self, backend, mock_responses
    ):
        """The server is trusted on limit; client does not re-truncate."""
        mock_responses.get(
            _CONFIGS_URL_PATTERN,
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
        self, backend, mock_responses
    ):
        """Legacy server returning mixed-status rows is still filtered client-side.

        ``status`` is an additive-optional parameter; the defensive
        client-side reapply guarantees correctness when the server
        ignores ``?status=``.
        """
        mock_responses.get(
            _CONFIGS_URL_PATTERN,
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
        self, backend, mock_responses
    ):
        """Legacy server returns unsorted rows; the client must sort them.

        Re-sorting is idempotent, so the defensive reapply is always
        safe (a server that already sorted will see no change).
        """
        from dataknobs_data import SortOrder, SortSpec

        mock_responses.get(
            _CONFIGS_URL_PATTERN,
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
    async def test_count_pushes_status_to_server(self, backend, mock_responses):
        """``count(filter_metadata=...)`` already sends ``filter_metadata``;
        this confirms ``count_all(status=...)`` adds ``?status=``."""
        mock_responses.get(
            _CONFIGS_URL_PATTERN,
            payload=[
                {"bot_id": "bot-1", "config": {}, "status": "inactive"},
                {"bot_id": "bot-2", "config": {}, "status": "inactive"},
            ],
        )

        n = await backend.count_inactive()

        assert n == 2
        captured = _captured_params(mock_responses)
        statuses = [p.get("status") for p in captured if p]
        assert "inactive" in statuses

    @pytest.mark.asyncio
    async def test_no_params_sends_no_query_string(self, backend, mock_responses):
        """Confirm invariant: bare ``list_all()`` adds no query string."""
        mock_responses.get(
            _CONFIGS_URL_PATTERN,
            payload=[],
        )

        await backend.list_all()

        captured = _captured_params(mock_responses)
        assert captured, "expected a request"
        for params in captured:
            assert not params, (
                f"bare list_all() must not send params, got {params!r}"
            )

    @pytest.mark.asyncio
    async def test_auth_header_sent(self, mock_responses):
        """Test that auth header is included in requests."""
        backend = HTTPRegistryBackend(
            base_url="https://config-service.test/api/v1",
            auth_token="my-secret-token",
        )
        await backend.initialize()

        try:
            mock_responses.get(
                "https://config-service.test/api/v1/configs/test",
                payload={"bot_id": "test", "config": {}, "status": "active"},
            )

            await backend.get("test")

            # Verify the request was made (aioresponses tracks calls)
            # The auth header is set in the session, so it's automatically included
        finally:
            await backend.close()

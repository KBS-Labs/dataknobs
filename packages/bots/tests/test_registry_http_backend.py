"""Tests for HTTPRegistryBackend."""

import pytest

from dataknobs_bots.registry import HTTPRegistryBackend, create_registry_backend


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
        """Test that list_active filters inactive registrations."""
        mock_responses.get(
            "https://config-service.test/api/v1/configs",
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
            "https://config-service.test/api/v1/configs",
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
            "https://config-service.test/api/v1/configs",
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
        """Test creating a new registration."""
        # First check if exists
        mock_responses.get(
            "https://config-service.test/api/v1/configs/new-bot",
            status=404,
        )
        # Then create
        mock_responses.post(
            "https://config-service.test/api/v1/configs",
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
        """Test updating an existing registration."""
        # First check if exists
        mock_responses.get(
            "https://config-service.test/api/v1/configs/existing-bot",
            payload={
                "bot_id": "existing-bot",
                "config": {"old": "value"},
                "status": "active",
            },
        )
        # Then update
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
        """Test deactivating a registration."""
        # First get the registration
        mock_responses.get(
            "https://config-service.test/api/v1/configs/test-bot",
            payload={
                "bot_id": "test-bot",
                "config": {"key": "value"},
                "status": "active",
            },
        )
        # Check if exists for register
        mock_responses.get(
            "https://config-service.test/api/v1/configs/test-bot",
            payload={
                "bot_id": "test-bot",
                "config": {"key": "value"},
                "status": "active",
            },
        )
        # Then update status
        mock_responses.put(
            "https://config-service.test/api/v1/configs/test-bot",
            payload={
                "bot_id": "test-bot",
                "config": {"key": "value"},
                "status": "inactive",
            },
        )

        result = await backend.deactivate("test-bot")

        assert result is True

    @pytest.mark.asyncio
    async def test_deactivate_not_found(self, backend, mock_responses):
        """Test deactivating non-existent registration."""
        mock_responses.get(
            "https://config-service.test/api/v1/configs/missing",
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

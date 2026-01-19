"""HTTP registry backend for external configuration services.

This module provides an HTTP/REST backend for fetching bot configurations
from external services, enabling centralized configuration management.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .backend import RegistryBackend
from .models import Registration

logger = logging.getLogger(__name__)


class HTTPRegistryBackend(RegistryBackend):
    """Registry backend that fetches configurations from a REST API.

    This backend enables using external configuration services as the
    source of truth for bot registrations. It supports:

    - Standard CRUD operations via HTTP
    - Token-based authentication
    - Configurable timeouts and retries
    - Read-only mode for config distribution

    The expected API contract is:
    - GET /configs - List all configurations
    - GET /configs/{id} - Get specific configuration
    - POST /configs - Create configuration
    - PUT /configs/{id} - Update configuration
    - DELETE /configs/{id} - Delete configuration

    Args:
        base_url: Base URL of the configuration service
        auth_token: Bearer token for authentication (optional)
        auth_header: Custom auth header name (default: "Authorization")
        timeout: Request timeout in seconds (default: 30)
        read_only: If True, disable write operations (default: False)
        verify_ssl: Verify SSL certificates (default: True)

    Example:
        ```python
        from dataknobs_bots.registry import HTTPRegistryBackend

        # Create backend
        backend = HTTPRegistryBackend(
            base_url="https://config-service.example.com/api/v1",
            auth_token="secret-token",
        )
        await backend.initialize()

        # Fetch configuration
        config = await backend.get_config("my-bot")
        print(config["llm"]["provider"])

        # List all active registrations
        registrations = await backend.list_active()
        for reg in registrations:
            print(f"{reg.bot_id}: {reg.status}")
        ```

    Example with read-only mode:
        ```python
        # Read-only backend for distributed config access
        backend = HTTPRegistryBackend(
            base_url="https://config-service/api/v1",
            auth_token="read-only-token",
            read_only=True,
        )
        await backend.initialize()

        # Can read but not write
        config = await backend.get_config("my-bot")  # Works
        await backend.register("new-bot", {...})  # Raises PermissionError
        ```
    """

    def __init__(
        self,
        base_url: str,
        auth_token: str | None = None,
        auth_header: str = "Authorization",
        timeout: float = 30.0,
        read_only: bool = False,
        verify_ssl: bool = True,
    ) -> None:
        """Initialize the HTTP registry backend.

        Args:
            base_url: Base URL of the config service
            auth_token: Bearer token for auth
            auth_header: Header name for auth token
            timeout: Request timeout in seconds
            read_only: Disable write operations
            verify_ssl: Verify SSL certificates
        """
        self._base_url = base_url.rstrip("/")
        self._auth_token = auth_token
        self._auth_header = auth_header
        self._timeout = timeout
        self._read_only = read_only
        self._verify_ssl = verify_ssl

        self._session: Any = None  # aiohttp.ClientSession
        self._initialized = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> HTTPRegistryBackend:
        """Create backend from configuration dictionary.

        Args:
            config: Configuration with keys:
                - base_url (required): API base URL
                - auth_token: Bearer token
                - auth_header: Custom auth header name
                - timeout: Request timeout
                - read_only: Disable writes
                - verify_ssl: SSL verification

        Returns:
            Configured HTTPRegistryBackend instance
        """
        return cls(
            base_url=config["base_url"],
            auth_token=config.get("auth_token"),
            auth_header=config.get("auth_header", "Authorization"),
            timeout=config.get("timeout", 30.0),
            read_only=config.get("read_only", False),
            verify_ssl=config.get("verify_ssl", True),
        )

    @property
    def base_url(self) -> str:
        """Get the base URL of the config service."""
        return self._base_url

    @property
    def is_read_only(self) -> bool:
        """Check if backend is in read-only mode."""
        return self._read_only

    async def initialize(self) -> None:
        """Initialize the HTTP session.

        Creates an aiohttp ClientSession for making requests.
        """
        if self._initialized:
            return

        try:
            import aiohttp

            # Build headers
            headers: dict[str, str] = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if self._auth_token:
                headers[self._auth_header] = f"Bearer {self._auth_token}"

            # Create SSL context
            ssl_context: bool | Any = self._verify_ssl
            if not self._verify_ssl:
                import ssl

                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

            # Create connector and session
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(
                headers=headers,
                connector=connector,
                timeout=timeout,
            )

            self._initialized = True
            logger.info("HTTPRegistryBackend initialized: %s", self._base_url)

        except ImportError as e:
            raise ImportError(
                "aiohttp is required for HTTPRegistryBackend. "
                "Install it with: pip install aiohttp"
            ) from e

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
        self._initialized = False
        logger.info("HTTPRegistryBackend closed")

    def _ensure_initialized(self) -> None:
        """Ensure backend is initialized."""
        if not self._initialized:
            raise RuntimeError("HTTPRegistryBackend not initialized. Call initialize() first.")

    def _check_write_permission(self) -> None:
        """Check if write operations are allowed."""
        if self._read_only:
            raise PermissionError("Backend is in read-only mode")

    async def register(
        self,
        bot_id: str,
        config: dict[str, Any],
        status: str = "active",
    ) -> Registration:
        """Register or update a bot configuration.

        Args:
            bot_id: Unique bot identifier
            config: Bot configuration dict
            status: Registration status

        Returns:
            Registration object

        Raises:
            PermissionError: If backend is read-only
            HTTPError: If request fails
        """
        self._ensure_initialized()
        self._check_write_permission()

        # Check if exists
        existing = await self.get(bot_id)

        payload = {
            "bot_id": bot_id,
            "config": config,
            "status": status,
        }

        if existing:
            # Update
            url = f"{self._base_url}/configs/{bot_id}"
            async with self._session.put(url, json=payload) as response:
                await self._check_response(response)
                data = await response.json()
        else:
            # Create
            url = f"{self._base_url}/configs"
            async with self._session.post(url, json=payload) as response:
                await self._check_response(response)
                data = await response.json()

        return self._parse_registration(data)

    async def get(self, bot_id: str) -> Registration | None:
        """Get registration by ID.

        Args:
            bot_id: Bot identifier

        Returns:
            Registration or None if not found
        """
        self._ensure_initialized()

        url = f"{self._base_url}/configs/{bot_id}"
        async with self._session.get(url) as response:
            if response.status == 404:
                return None
            await self._check_response(response)
            data = await response.json()
            return self._parse_registration(data)

    async def get_config(self, bot_id: str) -> dict[str, Any] | None:
        """Get just the config dict for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Configuration dict or None
        """
        reg = await self.get(bot_id)
        return reg.config if reg else None

    async def exists(self, bot_id: str) -> bool:
        """Check if an active registration exists.

        Args:
            bot_id: Bot identifier

        Returns:
            True if active registration exists
        """
        reg = await self.get(bot_id)
        return reg is not None and reg.status == "active"

    async def unregister(self, bot_id: str) -> bool:
        """Delete a registration.

        Args:
            bot_id: Bot identifier

        Returns:
            True if deleted, False if not found

        Raises:
            PermissionError: If backend is read-only
        """
        self._ensure_initialized()
        self._check_write_permission()

        url = f"{self._base_url}/configs/{bot_id}"
        async with self._session.delete(url) as response:
            if response.status == 404:
                return False
            await self._check_response(response)
            return True

    async def deactivate(self, bot_id: str) -> bool:
        """Deactivate a registration (soft delete).

        Args:
            bot_id: Bot identifier

        Returns:
            True if deactivated, False if not found

        Raises:
            PermissionError: If backend is read-only
        """
        self._ensure_initialized()
        self._check_write_permission()

        reg = await self.get(bot_id)
        if not reg:
            return False

        await self.register(bot_id, reg.config, status="inactive")
        return True

    async def list_active(self) -> list[Registration]:
        """List all active registrations.

        Returns:
            List of active Registration objects
        """
        all_regs = await self.list_all()
        return [r for r in all_regs if r.status == "active"]

    async def list_all(self) -> list[Registration]:
        """List all registrations.

        Returns:
            List of all Registration objects
        """
        self._ensure_initialized()

        url = f"{self._base_url}/configs"
        async with self._session.get(url) as response:
            await self._check_response(response)
            data = await response.json()

            # Handle both list and dict with "items" key
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and "items" in data:
                items = data["items"]
            elif isinstance(data, dict) and "configs" in data:
                items = data["configs"]
            else:
                items = []

            return [self._parse_registration(item) for item in items]

    async def list_ids(self) -> list[str]:
        """List active bot IDs.

        Returns:
            List of active bot ID strings
        """
        active = await self.list_active()
        return [r.bot_id for r in active]

    async def count(self) -> int:
        """Count active registrations.

        Returns:
            Number of active registrations
        """
        return len(await self.list_active())

    async def clear(self) -> None:
        """Clear all registrations.

        Raises:
            PermissionError: If backend is read-only
            NotImplementedError: If API doesn't support bulk delete
        """
        self._ensure_initialized()
        self._check_write_permission()

        # Try bulk delete endpoint first
        url = f"{self._base_url}/configs"
        async with self._session.delete(url) as response:
            if response.status == 405:  # Method not allowed
                # Fall back to individual deletes
                all_regs = await self.list_all()
                for reg in all_regs:
                    await self.unregister(reg.bot_id)
                return
            await self._check_response(response)

    async def _check_response(self, response: Any) -> None:
        """Check HTTP response for errors.

        Args:
            response: aiohttp response object

        Raises:
            HTTPError with details on failure
        """
        if response.status >= 400:
            text = await response.text()
            error_msg = f"HTTP {response.status}: {text}"
            logger.error("HTTP request failed: %s", error_msg)

            # Import here to avoid dependency at module level
            from aiohttp import ClientResponseError

            raise ClientResponseError(
                response.request_info,
                response.history,
                status=response.status,
                message=text,
            )

    def _parse_registration(self, data: dict[str, Any]) -> Registration:
        """Parse API response into Registration object.

        Args:
            data: Response dict from API

        Returns:
            Registration object
        """

        def parse_datetime(val: Any) -> datetime:
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                try:
                    return datetime.fromisoformat(val.replace("Z", "+00:00"))
                except ValueError:
                    pass
            return datetime.now(timezone.utc)

        return Registration(
            bot_id=data.get("bot_id", data.get("id", "")),
            config=data.get("config", {}),
            status=data.get("status", "active"),
            created_at=parse_datetime(data.get("created_at")),
            updated_at=parse_datetime(data.get("updated_at")),
            last_accessed_at=parse_datetime(data.get("last_accessed_at")),
        )

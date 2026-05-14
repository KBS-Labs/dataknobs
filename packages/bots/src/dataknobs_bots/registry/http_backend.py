"""HTTP registry backend for external configuration services.

This module provides an HTTP/REST backend for fetching bot configurations
from external services, enabling centralized configuration management.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .backend import RegistryBackend
from .memory import _apply_sort_limit_offset, _matches_metadata
from .models import Registration

if TYPE_CHECKING:
    from dataknobs_data import SortSpec, StreamConfig

logger = logging.getLogger(__name__)

# Seconds to sleep after aiohttp ClientSession.close() so that SSL transport
# callbacks can drain before event loop shutdown.  See dk-29 for full context.
_AIOHTTP_DRAIN_SECS = 0.25


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
            await asyncio.sleep(_AIOHTTP_DRAIN_SECS)
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
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Registration:
        """Register or update a bot configuration.

        Issues a single ``PUT /configs/{bot_id}`` — the server's PUT
        handler routes to ``RegistryBackend.register`` which is
        upsert-style on every implementation (it creates a new
        registration if one doesn't exist, updates it if it does).
        This avoids the prior touching GET that probed for existence
        and corrupted ``last_accessed_at`` on every re-register.

        Args:
            bot_id: Unique bot identifier
            config: Bot configuration dict
            status: Registration status
            metadata: Cross-cutting context (tenant_id, audit, feature
                flags). Included in the PUT JSON body as an optional
                ``metadata`` field; servers that don't honor it
                ignore it (additive contract).

        Returns:
            Registration object

        Raises:
            PermissionError: If backend is read-only
            HTTPError: If request fails
        """
        self._ensure_initialized()
        self._check_write_permission()

        payload: dict[str, Any] = {
            "bot_id": bot_id,
            "config": config,
            "status": status,
            "metadata": dict(metadata or {}),
        }

        url = f"{self._base_url}/configs/{bot_id}"
        async with self._session.put(url, json=payload) as response:
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

        Server-side activity-tracking semantics apply. This client
        does not maintain client-side ``last_accessed_at`` state, so
        whether this read registers as user activity is determined
        entirely by the remote server's contract. :meth:`peek_config`
        intentionally has the same wire behavior (see its docstring
        for the rationale).

        Args:
            bot_id: Bot identifier

        Returns:
            Configuration dict or None
        """
        reg = await self.get(bot_id)
        return reg.config if reg else None

    async def peek_config(self, bot_id: str) -> dict[str, Any] | None:
        """Get config without client-side activity-tracking mutation.

        ``HTTPRegistryBackend`` does not maintain client-side
        ``last_accessed_at`` state — the server controls activity
        tracking via its own contract. The Protocol's non-mutation
        guarantee is therefore satisfied at the client surface
        unconditionally; behavior is identical to :meth:`get_config`.

        Servers that distinguish ``GET /configs/{id}`` from a
        non-touching peek MAY surface that distinction via header,
        query parameter, or sibling endpoint — that is a server-side
        contract. This client deliberately does not impose a
        wire-protocol distinction.

        Args:
            bot_id: Bot identifier

        Returns:
            Config dict if found, None otherwise
        """
        return await self.get_config(bot_id)

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

        Issues ``POST /configs/{bot_id}/deactivate`` — a dedicated
        endpoint that routes to ``RegistryBackend.deactivate`` on the
        server.  Avoids the prior touching read (``self.get`` + then
        ``self.register(..., status="inactive")``) that bumped
        ``last_accessed_at`` on every soft-delete, which contradicted
        the user-activity signal that timestamp is supposed to carry.

        Args:
            bot_id: Bot identifier

        Returns:
            True if deactivated, False if not found

        Raises:
            PermissionError: If backend is read-only
        """
        self._ensure_initialized()
        self._check_write_permission()

        url = f"{self._base_url}/configs/{bot_id}/deactivate"
        async with self._session.post(url) as response:
            if response.status == 404:
                return False
            await self._check_response(response)
            return True

    async def list_active(
        self,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
        sort: list[SortSpec] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Registration]:
        """List all active registrations.

        Convenience wrapper for :meth:`list_all` with ``status="active"``.
        All filter parameters are pushed to the server via the
        ``GET /configs`` query string (see :meth:`list_all` for the
        wire-protocol contract).
        """
        return await self.list_all(
            status="active",
            filter_metadata=filter_metadata,
            sort=sort,
            limit=limit,
            offset=offset,
        )

    async def list_inactive(
        self,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
        sort: list[SortSpec] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Registration]:
        """List all inactive registrations.

        Symmetric counterpart to :meth:`list_active`.
        """
        return await self.list_all(
            status="inactive",
            filter_metadata=filter_metadata,
            sort=sort,
            limit=limit,
            offset=offset,
        )

    async def list_all(
        self,
        *,
        status: str | None = None,
        filter_metadata: Mapping[str, Any] | None = None,
        sort: list[SortSpec] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Registration]:
        """List registrations matching the supplied filters.

        Wire protocol:
            Filter parameters are pushed to the server on
            ``GET /configs`` as the following query parameters
            (all optional, omitted when ``None``):

            - ``?filter_metadata=<URL-encoded JSON object>`` —
              ``filter_metadata`` is serialized with
              ``sort_keys=True`` so the query string is deterministic
              (server cache keys / request logs).
            - ``?status=<value>`` — equality filter on the ``status``
              column.
            - ``?sort=<field>[:asc|desc]`` (repeatable) — multi-key
              sort spec, encoded as ``field:order`` with the
              lowercase :class:`SortOrder` value. Wire order of
              repeated values is preserved as the tie-break order.
            - ``?limit=<int>`` — row limit.
            - ``?offset=<int>`` — row offset for pagination.

            The server-side schema is **additive optional**: servers
            that recognize a parameter SHOULD honor it; servers that
            don't ignore it and return the broader, unfiltered list.
            To keep correctness on legacy servers, the client
            defensively re-applies idempotent filters
            (``filter_metadata``, ``status``, ``sort``) after parsing
            the response.

            ``limit`` and ``offset`` are **not** re-applied
            client-side — that would corrupt server-side pagination
            (re-offsetting an already-offset window drops live rows).
            Servers that don't honor pagination return the full
            list, and the caller iterates more rows than requested.
            This is the documented degradation mode for
            non-conforming servers.

        Args:
            status: Optional equality filter on the ``status`` field.
                ``None`` returns all statuses.
            filter_metadata: Optional equality filter over the
                ``metadata`` channel. ``None`` / empty mapping sends
                no query parameter.
            sort: Optional sort specification.
            limit: Optional row limit.
            offset: Optional row offset for pagination.

        Returns:
            List of matching Registration objects.
        """
        self._ensure_initialized()

        url = f"{self._base_url}/configs"
        params: dict[str, Any] = {}
        if filter_metadata:
            params["filter_metadata"] = json.dumps(
                dict(filter_metadata), sort_keys=True
            )
        if status is not None:
            params["status"] = status
        if sort:
            # Repeated query param — aiohttp handles list values by
            # serializing one ``?sort=`` per entry, preserving order.
            params["sort"] = [
                f"{spec.field}:{spec.order.value}" for spec in sort
            ]
        if limit is not None:
            params["limit"] = str(limit)
        if offset is not None:
            params["offset"] = str(offset)
        actual_params = params if params else None

        async with self._session.get(url, params=actual_params) as response:
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

            regs = [self._parse_registration(item) for item in items]
            # Defensive client-side reapply of idempotent filters
            # for additive-optional legacy servers.  See class docstring
            # for the policy split — limit/offset are intentionally
            # NOT reapplied (corruption risk on server-paginated
            # responses).
            if filter_metadata:
                regs = [r for r in regs if _matches_metadata(r, filter_metadata)]
            if status is not None:
                regs = [r for r in regs if r.status == status]
            if sort:
                regs = _apply_sort_limit_offset(
                    regs, sort, limit=None, offset=None
                )
            return regs

    async def list_ids(self) -> list[str]:
        """List active bot IDs.

        Returns:
            List of active bot ID strings
        """
        active = await self.list_active()
        return [r.bot_id for r in active]

    async def count_all(
        self,
        *,
        status: str | None = None,
        filter_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Count registrations, optionally filtered by status and metadata.

        Pushes ``filter_metadata`` to the server via :meth:`list_all`;
        ``status`` is applied client-side after the response is parsed.

        ``limit`` and ``offset`` are intentionally NOT passed to
        :meth:`list_all` here — the count must reflect the full
        matching set, not a paginated window.  The correctness of
        ``len(list_all(...))`` relies on :meth:`list_all` defensively
        reapplying ``filter_metadata`` and ``status`` client-side, so
        a non-conforming server that returned the unfiltered list
        still produces the correct count.  If the defensive-reapply
        policy on :meth:`list_all` ever changes to skip
        ``filter_metadata`` / ``status``, this method must be updated
        to issue a server-side count instead.
        """
        return len(
            await self.list_all(
                status=status, filter_metadata=filter_metadata
            )
        )

    async def count(
        self,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Count active registrations.

        Convenience for :meth:`count_all` with ``status="active"``.
        """
        return await self.count_all(
            status="active", filter_metadata=filter_metadata
        )

    async def count_inactive(
        self,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Count inactive registrations.

        Convenience for :meth:`count_all` with ``status="inactive"``.
        """
        return await self.count_all(
            status="inactive", filter_metadata=filter_metadata
        )

    async def stream(
        self,
        *,
        status: str | None = None,
        filter_metadata: Mapping[str, Any] | None = None,
        config: StreamConfig | None = None,
    ) -> AsyncIterator[Registration]:
        """Stream registrations matching the supplied filters.

        The HTTP wire today does not support streaming responses, so
        this method fetches all matching registrations via
        :meth:`list_all` and yields them sequentially.  ``config`` is
        accepted for protocol compatibility and ignored.
        """
        del config  # No batching needed for the HTTP backend today.
        regs = await self.list_all(status=status, filter_metadata=filter_metadata)
        for reg in regs:
            yield reg

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
            metadata=dict(data.get("metadata") or {}),
            created_at=parse_datetime(data.get("created_at")),
            updated_at=parse_datetime(data.get("updated_at")),
            last_accessed_at=parse_datetime(data.get("last_accessed_at")),
        )

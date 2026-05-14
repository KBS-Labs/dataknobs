"""Reference HTTP router exposing :class:`RegistryBackend` as the wire protocol.

:class:`HTTPRegistryBackend` speaks a small REST contract over
``/configs``.  Without a matching server, every consumer that uses the
HTTP backend has to write the same handful of route handlers by hand —
and any drift between client expectations and server implementation
breaks silently at the wire.

:func:`create_registry_router` mounts the same protocol as a FastAPI
router so consumers can stand up a config service with one line of
glue::

    from fastapi import FastAPI
    from dataknobs_bots.registry import (
        DataKnobsRegistryAdapter,
        create_registry_router,
    )

    app = FastAPI()
    adapter = DataKnobsRegistryAdapter(
        backend_type="postgres",
        backend_config={"host": "...", "database": "..."},
    )

    @app.on_event("startup")
    async def startup() -> None:
        await adapter.initialize()
        app.include_router(
            create_registry_router(adapter),
            prefix="/api/v1",
        )

The protocol is pinned on both sides of the wire — the client side by
``tests/test_registry_http_backend.py`` (what the client sends) and the
server side by ``tests/test_registry_server.py`` (what the server
accepts and returns).  Drift in either direction breaks both test
suites.

Efficiency note:
    When the underlying backend is :class:`DataKnobsRegistryAdapter`
    over Postgres, the ``filter_metadata`` query parameter is pushed
    down to a ``metadata->>'<key>' = $1`` predicate (text extraction
    + equality) — server-side filtering rather than client-side
    scanning, but **not** an indexed lookup by default.  Postgres's
    ``auto_create_table=True`` path creates a GIN index over the
    ``metadata`` JSONB column using the default ``jsonb_ops``
    opclass, which optimizes containment / existence operators
    (``@>``, ``?``, ``?&``, ``?|``) and **does not** accelerate
    ``->>``-style text extractions.  Consumers who need O(index
    lookup) on a hot ``filter_metadata`` key (e.g. ``tenant_id``)
    should add an expression index in their own migrations::

        CREATE INDEX idx_bot_configs_tenant_id
          ON bot_configs ((metadata->>'tenant_id'));

    Backends without a JSONB column fall back to per-row matching via
    the ``metadata.X`` field-path translation in their respective
    query translators; this is also correct but unindexed.  See
    ``packages/bots/docs/DYNAMIC_REGISTRATION.md`` for the full
    contract.

FastAPI is treated as an optional dependency: importing this module
without FastAPI installed succeeds, but calling
:func:`create_registry_router` raises ``ImportError`` with an install
hint (``pip install 'dataknobs-bots[server]'``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from dataknobs_data import SortOrder, SortSpec

from .backend import RegistryBackend

if TYPE_CHECKING:
    from fastapi import APIRouter

logger = logging.getLogger(__name__)


def _parse_filter_metadata(filter_metadata: str | None) -> Mapping[str, Any] | None:
    """Decode the ``?filter_metadata=`` query parameter to a dict.

    Returns ``None`` for ``None``/empty string.  Raises FastAPI
    ``HTTPException(422)`` if the value is present but not a JSON
    object — mirrors the client-side contract documented on
    :meth:`HTTPRegistryBackend.list_all` that the parameter is always
    JSON-encoded.
    """
    if not filter_metadata:
        return None
    try:
        decoded = json.loads(filter_metadata)
    except json.JSONDecodeError as exc:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=422,
            detail=f"filter_metadata must be valid JSON: {exc.msg}",
        ) from exc
    if not isinstance(decoded, dict):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=422,
            detail="filter_metadata must be a JSON object",
        )
    return decoded


def _parse_sort(sort: list[str] | None) -> list[SortSpec] | None:
    """Decode repeated ``?sort=<field>[:asc|desc]`` query params to a list of SortSpec.

    The wire format is ``field:order`` where ``order`` is the
    lowercase :class:`SortOrder` value (``"asc"`` or ``"desc"``).
    A bare field name (no ``:order``) defaults to ``"asc"`` —
    consistent with :class:`SortSpec`'s own default.

    The list order is preserved end-to-end: ``?sort=a:asc&sort=b:desc``
    produces ``[SortSpec("a", ASC), SortSpec("b", DESC)]`` so multi-key
    sort tie-breaking matches the caller's intent.

    Returns ``None`` for ``None`` / empty list (no-sort).  Raises
    FastAPI ``HTTPException(422)`` on malformed entries — empty
    field, unknown order, etc.
    """
    if not sort:
        return None
    from fastapi import HTTPException

    specs: list[SortSpec] = []
    for entry in sort:
        if ":" in entry:
            field, order_str = entry.split(":", 1)
        else:
            field, order_str = entry, "asc"
        if not field:
            raise HTTPException(
                status_code=422,
                detail=f"sort entry {entry!r} has an empty field",
            )
        order_norm = order_str.lower()
        if order_norm not in ("asc", "desc"):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"sort entry {entry!r} has invalid order {order_str!r}; "
                    "must be 'asc' or 'desc'"
                ),
            )
        specs.append(
            SortSpec(
                field=field,
                order=SortOrder.ASC if order_norm == "asc" else SortOrder.DESC,
            )
        )
    return specs


def create_registry_router(backend: RegistryBackend) -> APIRouter:
    """Build a FastAPI router exposing ``backend`` as the registry wire protocol.

    Routes mounted relative to the router's prefix:

    - ``GET    /configs``           — list all registrations
      Query (all optional):

      - ``filter_metadata=<URL-encoded JSON object>`` — equality
        filter over the ``metadata`` channel.
      - ``status=active|inactive`` — equality filter on the
        ``status`` column. ``None`` returns all statuses.
      - ``sort=<field>[:asc|desc]`` (repeatable) — multi-key sort
        spec.  Order defaults to ``asc`` when omitted; the wire
        order of repeated values is preserved as the sort
        tie-break order.
      - ``limit=<int>`` (``>=0``) — row limit applied after sort.
      - ``offset=<int>`` (``>=0``) — row offset applied before
        ``limit``.

      Response 200: ``{"items": [<registration dict>, ...]}``.
    - ``GET    /configs/{bot_id}``  — get one
      Response 200: registration dict, 404 if not found.
    - ``POST   /configs``           — create (or upsert)
      Body: ``{"bot_id": str, "config": dict, "status"?: str, "metadata"?: dict}``.
      Response 200: registration dict.
    - ``PUT    /configs/{bot_id}``  — update (or upsert)
      Body: ``{"config": dict, "status"?: str, "metadata"?: dict}``.
      Response 200: registration dict.
    - ``DELETE /configs/{bot_id}``  — hard delete
      Response 204 on success, 404 if not found.
    - ``DELETE /configs``           — clear all
      Response 204.

    The router does not call ``initialize()`` on the backend — the
    caller owns lifecycle.  Caller is also responsible for
    authentication, rate limiting, and tenant isolation; this router
    is the protocol surface, not a security boundary.

    Args:
        backend: An initialized :class:`RegistryBackend` implementation.

    Returns:
        A FastAPI ``APIRouter`` ready to mount via ``app.include_router``.

    Raises:
        ImportError: If FastAPI is not installed.
    """
    try:
        from fastapi import APIRouter, HTTPException, Query, Response
    except ImportError as exc:
        raise ImportError(
            "fastapi is required for create_registry_router. "
            "Install with: pip install 'dataknobs-bots[server]'"
        ) from exc

    router = APIRouter()

    @router.get("/configs")
    async def list_configs(
        filter_metadata: str | None = Query(default=None),
        status: str | None = Query(default=None),
        # B008 flags ``Query()`` only for list-typed defaults; bugbear's
        # rule fires on the mutable container type, not the call itself
        # (FastAPI's ``Query`` is on the standard allowlist).  The
        # ``Annotated[list[str] | None, Query()]`` form would silence it
        # cleanly, but ``from __future__ import annotations`` at this
        # module's top turns it into a forward reference that FastAPI /
        # pydantic can't resolve at route-registration time.  Plain
        # default + ``noqa`` is the path of least surprise.
        sort: list[str] | None = Query(default=None),  # noqa: B008
        limit: int | None = Query(default=None, ge=0),
        offset: int | None = Query(default=None, ge=0),
    ) -> dict[str, Any]:
        decoded_filter = _parse_filter_metadata(filter_metadata)
        parsed_sort = _parse_sort(sort)
        regs = await backend.list_all(
            status=status,
            filter_metadata=decoded_filter,
            sort=parsed_sort,
            limit=limit,
            offset=offset,
        )
        return {"items": [r.to_dict() for r in regs]}

    @router.get("/configs/{bot_id}")
    async def get_config(bot_id: str) -> dict[str, Any]:
        reg = await backend.get(bot_id)
        if reg is None:
            raise HTTPException(
                status_code=404,
                detail=f"Bot {bot_id!r} not found",
            )
        return reg.to_dict()

    @router.post("/configs")
    async def create_config(payload: dict[str, Any]) -> dict[str, Any]:
        if "bot_id" not in payload or "config" not in payload:
            raise HTTPException(
                status_code=422,
                detail="Body must include 'bot_id' and 'config'",
            )
        reg = await backend.register(
            payload["bot_id"],
            payload["config"],
            status=payload.get("status", "active"),
            metadata=payload.get("metadata"),
        )
        return reg.to_dict()

    @router.put("/configs/{bot_id}")
    async def update_config(
        bot_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if "config" not in payload:
            raise HTTPException(
                status_code=422,
                detail="Body must include 'config'",
            )
        reg = await backend.register(
            bot_id,
            payload["config"],
            status=payload.get("status", "active"),
            metadata=payload.get("metadata"),
        )
        return reg.to_dict()

    @router.delete("/configs/{bot_id}")
    async def delete_config(bot_id: str) -> Response:
        deleted = await backend.unregister(bot_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Bot {bot_id!r} not found",
            )
        return Response(status_code=204)

    @router.post("/configs/{bot_id}/deactivate")
    async def deactivate_config(bot_id: str) -> Response:
        """Soft-delete a bot via the backend's :meth:`deactivate` method.

        Dedicated endpoint (rather than reusing PUT with
        ``status="inactive"``) so the HTTP client can deactivate
        without first issuing a touching ``GET /configs/{bot_id}`` to
        recover the config payload required by PUT.
        """
        deactivated = await backend.deactivate(bot_id)
        if not deactivated:
            raise HTTPException(
                status_code=404,
                detail=f"Bot {bot_id!r} not found",
            )
        return Response(status_code=204)

    @router.delete("/configs")
    async def clear_configs() -> Response:
        await backend.clear()
        return Response(status_code=204)

    return router

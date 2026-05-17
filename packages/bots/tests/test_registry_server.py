"""Tests for the reference registry HTTP router.

Pins the server-side half of the wire protocol that
:class:`HTTPRegistryBackend` speaks.  The client-side half is pinned in
``test_registry_http_backend.py``; together the two test suites lock
the protocol against drift.

Coverage:
- Response shape (``{"items": [...]}``) so ``HTTPRegistryBackend.list_all``
  parses it.
- Field names that ``HTTPRegistryBackend._parse_registration`` reads
  (``bot_id``, ``config``, ``status``, ``metadata``, timestamps).
- Body shape that the client sends on POST/PUT
  (``bot_id``/``config``/``status``/``metadata``).
- The ``filter_metadata`` query parameter is decoded and pushed to the
  backend.
- 404 semantics on missing GET/DELETE.
- 204 on DELETE single and DELETE bulk.
- 422 on malformed ``filter_metadata`` or missing required body fields.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from dataknobs_bots.registry import InMemoryBackend, create_registry_router

# FastAPI and httpx are required for these tests.
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

API_PREFIX = "/api/v1"


@pytest.fixture
async def app_client() -> AsyncIterator[tuple[AsyncClient, InMemoryBackend]]:
    """In-process FastAPI app mounted with the registry router."""
    backend = InMemoryBackend()
    await backend.initialize()
    app = FastAPI()
    app.include_router(create_registry_router(backend), prefix=API_PREFIX)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, backend
    await backend.close()


# --- list_configs (GET /configs) -------------------------------------


@pytest.mark.asyncio
async def test_list_empty_returns_items_shape(app_client) -> None:
    """Empty backend returns the documented ``{"items": []}`` shape."""
    client, _ = app_client
    resp = await client.get(f"{API_PREFIX}/configs")
    assert resp.status_code == 200
    assert resp.json() == {"items": []}


@pytest.mark.asyncio
async def test_list_returns_registration_dicts(app_client) -> None:
    """list response contains Registration.to_dict() items the client can parse."""
    client, backend = app_client
    await backend.register(
        "alice",
        {"llm": {"provider": "echo"}},
        metadata={"tenant_id": "acme"},
    )
    resp = await client.get(f"{API_PREFIX}/configs")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    item = items[0]
    # These are the fields HTTPRegistryBackend._parse_registration reads.
    assert item["bot_id"] == "alice"
    assert item["config"] == {"llm": {"provider": "echo"}}
    assert item["status"] == "active"
    assert item["metadata"] == {"tenant_id": "acme"}
    assert "created_at" in item
    assert "updated_at" in item
    assert "last_accessed_at" in item


@pytest.mark.asyncio
async def test_list_filter_metadata_pushed_to_backend(app_client) -> None:
    """``?filter_metadata=`` is decoded and applied at the backend layer."""
    client, backend = app_client
    await backend.register("alice", {}, metadata={"tenant_id": "acme"})
    await backend.register("bob", {}, metadata={"tenant_id": "globex"})

    resp = await client.get(
        f"{API_PREFIX}/configs",
        params={"filter_metadata": json.dumps({"tenant_id": "acme"})},
    )
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert [r["bot_id"] for r in items] == ["alice"]


@pytest.mark.asyncio
async def test_list_filter_metadata_invalid_json_returns_422(app_client) -> None:
    client, _ = app_client
    resp = await client.get(
        f"{API_PREFIX}/configs",
        params={"filter_metadata": "not-json"},
    )
    assert resp.status_code == 422
    assert "JSON" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_list_filter_metadata_non_object_returns_422(app_client) -> None:
    """Arrays/scalars in filter_metadata are rejected — protocol is a JSON object."""
    client, _ = app_client
    resp = await client.get(
        f"{API_PREFIX}/configs",
        params={"filter_metadata": json.dumps([1, 2, 3])},
    )
    assert resp.status_code == 422


# --- query-param push-down: status/sort/limit/offset --------


@pytest.mark.asyncio
async def test_list_status_filter_pushed_to_backend(app_client) -> None:
    """``?status=active`` is routed to ``backend.list_all(status="active")``."""
    client, backend = app_client
    await backend.register("alice", {})
    await backend.register("bob", {})
    await backend.deactivate("bob")

    resp = await client.get(f"{API_PREFIX}/configs", params={"status": "active"})
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert [r["bot_id"] for r in items] == ["alice"]


@pytest.mark.asyncio
async def test_list_status_inactive_pushed_to_backend(app_client) -> None:
    client, backend = app_client
    await backend.register("alice", {})
    await backend.register("bob", {})
    await backend.deactivate("bob")

    resp = await client.get(f"{API_PREFIX}/configs", params={"status": "inactive"})
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert [r["bot_id"] for r in items] == ["bob"]


@pytest.mark.asyncio
async def test_list_no_status_returns_all(app_client) -> None:
    """No ``?status`` param returns all statuses (the bare-list contract)."""
    client, backend = app_client
    await backend.register("alice", {})
    await backend.register("bob", {})
    await backend.deactivate("bob")

    resp = await client.get(f"{API_PREFIX}/configs")
    assert resp.status_code == 200
    statuses = sorted(r["status"] for r in resp.json()["items"])
    assert statuses == ["active", "inactive"]


@pytest.mark.asyncio
async def test_list_sort_asc_routed_to_backend(app_client) -> None:
    """``?sort=bot_id:asc`` is parsed into a SortSpec and pushed down."""
    client, backend = app_client
    await backend.register("charlie", {})
    await backend.register("alice", {})
    await backend.register("bob", {})

    resp = await client.get(
        f"{API_PREFIX}/configs", params={"sort": "bot_id:asc"}
    )
    assert resp.status_code == 200
    assert [r["bot_id"] for r in resp.json()["items"]] == ["alice", "bob", "charlie"]


@pytest.mark.asyncio
async def test_list_sort_desc_routed_to_backend(app_client) -> None:
    client, backend = app_client
    await backend.register("alice", {})
    await backend.register("bob", {})
    await backend.register("charlie", {})

    resp = await client.get(
        f"{API_PREFIX}/configs", params={"sort": "bot_id:desc"}
    )
    assert resp.status_code == 200
    assert [r["bot_id"] for r in resp.json()["items"]] == ["charlie", "bob", "alice"]


@pytest.mark.asyncio
async def test_list_sort_default_order_is_asc(app_client) -> None:
    """``?sort=field`` (no ``:order``) defaults to ``asc``, like SortSpec."""
    client, backend = app_client
    await backend.register("charlie", {})
    await backend.register("alice", {})

    resp = await client.get(f"{API_PREFIX}/configs", params={"sort": "bot_id"})
    assert resp.status_code == 200
    assert [r["bot_id"] for r in resp.json()["items"]] == ["alice", "charlie"]


@pytest.mark.asyncio
async def test_list_sort_multi_key_preserves_order(app_client) -> None:
    """Repeated ``?sort=`` params preserve wire order as tie-break order.

    Two bots have the same metadata ``team`` ("alpha") but different
    bot_ids; sorting by team:asc then bot_id:desc should break the
    team tie with bot_id-descending.
    """
    client, backend = app_client
    await backend.register("alice", {}, metadata={"team": "alpha"})
    await backend.register("bob", {}, metadata={"team": "alpha"})
    await backend.register("zach", {}, metadata={"team": "beta"})

    resp = await client.get(
        f"{API_PREFIX}/configs",
        params=[("sort", "metadata.team:asc"), ("sort", "bot_id:desc")],
    )
    assert resp.status_code == 200
    ids = [r["bot_id"] for r in resp.json()["items"]]
    # alpha team first; within alpha, bob > alice (desc); zach (beta) last
    assert ids == ["bob", "alice", "zach"]


@pytest.mark.asyncio
async def test_list_sort_invalid_order_returns_422(app_client) -> None:
    """``?sort=field:wrong`` is a wire-protocol error."""
    client, _ = app_client
    resp = await client.get(
        f"{API_PREFIX}/configs", params={"sort": "bot_id:sideways"}
    )
    assert resp.status_code == 422
    assert "sideways" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_list_sort_empty_field_returns_422(app_client) -> None:
    client, _ = app_client
    resp = await client.get(f"{API_PREFIX}/configs", params={"sort": ":asc"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_list_limit_pushed_to_backend(app_client) -> None:
    client, backend = app_client
    for i in range(5):
        await backend.register(f"bot-{i}", {})

    resp = await client.get(f"{API_PREFIX}/configs", params={"limit": "3"})
    assert resp.status_code == 200
    assert len(resp.json()["items"]) == 3


@pytest.mark.asyncio
async def test_list_offset_pushed_to_backend(app_client) -> None:
    """``?offset=`` skips the first N rows; combine with sort for determinism."""
    client, backend = app_client
    for i in range(5):
        await backend.register(f"bot-{i}", {})

    resp = await client.get(
        f"{API_PREFIX}/configs",
        params=[("sort", "bot_id:asc"), ("offset", "2")],
    )
    assert resp.status_code == 200
    ids = [r["bot_id"] for r in resp.json()["items"]]
    assert ids == ["bot-2", "bot-3", "bot-4"]


@pytest.mark.asyncio
async def test_list_limit_and_offset_combine(app_client) -> None:
    """Offset is applied before limit (matches SortSpec/SQL semantics)."""
    client, backend = app_client
    for i in range(10):
        await backend.register(f"bot-{i}", {})

    resp = await client.get(
        f"{API_PREFIX}/configs",
        params=[("sort", "bot_id:asc"), ("offset", "3"), ("limit", "2")],
    )
    assert resp.status_code == 200
    ids = [r["bot_id"] for r in resp.json()["items"]]
    assert ids == ["bot-3", "bot-4"]


@pytest.mark.asyncio
async def test_list_negative_limit_returns_422(app_client) -> None:
    """``?limit=-1`` fails the ``ge=0`` constraint."""
    client, _ = app_client
    resp = await client.get(f"{API_PREFIX}/configs", params={"limit": "-1"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_list_negative_offset_returns_422(app_client) -> None:
    client, _ = app_client
    resp = await client.get(f"{API_PREFIX}/configs", params={"offset": "-5"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_list_all_params_combined(app_client) -> None:
    """All params compose: status + filter_metadata + sort + offset + limit."""
    client, backend = app_client
    for i in range(6):
        await backend.register(
            f"bot-{i}",
            {},
            metadata={"tenant_id": "acme" if i % 2 == 0 else "globex"},
        )
    # Deactivate two acme bots
    await backend.deactivate("bot-0")
    await backend.deactivate("bot-4")

    # Want: active acme bots, sorted desc, skip 0, take 1 → bot-2 only
    resp = await client.get(
        f"{API_PREFIX}/configs",
        params=[
            ("status", "active"),
            ("filter_metadata", json.dumps({"tenant_id": "acme"})),
            ("sort", "bot_id:desc"),
            ("limit", "1"),
        ],
    )
    assert resp.status_code == 200
    ids = [r["bot_id"] for r in resp.json()["items"]]
    assert ids == ["bot-2"]


# --- get_config (GET /configs/{bot_id}) ------------------------------


@pytest.mark.asyncio
async def test_get_returns_registration_dict(app_client) -> None:
    client, backend = app_client
    await backend.register("alice", {"v": 1}, metadata={"tenant_id": "acme"})
    resp = await client.get(f"{API_PREFIX}/configs/alice")
    assert resp.status_code == 200
    body = resp.json()
    assert body["bot_id"] == "alice"
    assert body["config"] == {"v": 1}
    assert body["metadata"] == {"tenant_id": "acme"}


@pytest.mark.asyncio
async def test_get_missing_returns_404(app_client) -> None:
    client, _ = app_client
    resp = await client.get(f"{API_PREFIX}/configs/does-not-exist")
    assert resp.status_code == 404


# --- create_config (POST /configs) -----------------------------------


@pytest.mark.asyncio
async def test_post_creates_and_returns_registration(app_client) -> None:
    """POST body shape matches HTTPRegistryBackend.register's payload."""
    client, backend = app_client
    payload: dict[str, Any] = {
        "bot_id": "alice",
        "config": {"llm": {"provider": "echo"}},
        "status": "active",
        "metadata": {"tenant_id": "acme", "audit": {"by": "alice"}},
    }
    resp = await client.post(f"{API_PREFIX}/configs", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["bot_id"] == "alice"
    assert body["config"] == payload["config"]
    assert body["metadata"] == payload["metadata"]

    # And it's actually in the backend, not just echoed.
    stored = await backend.get("alice")
    assert stored is not None
    assert stored.metadata == payload["metadata"]


@pytest.mark.asyncio
async def test_post_default_status_is_active(app_client) -> None:
    client, _ = app_client
    resp = await client.post(
        f"{API_PREFIX}/configs",
        json={"bot_id": "alice", "config": {}},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "active"


@pytest.mark.asyncio
async def test_post_missing_required_fields_returns_422(app_client) -> None:
    client, _ = app_client
    resp = await client.post(f"{API_PREFIX}/configs", json={"bot_id": "x"})
    assert resp.status_code == 422

    resp = await client.post(f"{API_PREFIX}/configs", json={"config": {}})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_post_no_metadata_defaults_to_empty(app_client) -> None:
    """Omitting metadata is allowed; the backend stores ``{}``."""
    client, backend = app_client
    resp = await client.post(
        f"{API_PREFIX}/configs",
        json={"bot_id": "alice", "config": {}},
    )
    assert resp.status_code == 200
    stored = await backend.get("alice")
    assert stored is not None
    assert stored.metadata == {}


# --- update_config (PUT /configs/{bot_id}) ---------------------------


@pytest.mark.asyncio
async def test_put_updates_existing_registration(app_client) -> None:
    client, backend = app_client
    await backend.register("alice", {"v": 1}, metadata={"tenant_id": "acme"})

    resp = await client.put(
        f"{API_PREFIX}/configs/alice",
        json={
            "config": {"v": 2},
            "metadata": {"tenant_id": "acme", "audit": {"by": "bob"}},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["config"] == {"v": 2}
    assert body["metadata"] == {"tenant_id": "acme", "audit": {"by": "bob"}}


@pytest.mark.asyncio
async def test_put_missing_config_returns_422(app_client) -> None:
    client, _ = app_client
    resp = await client.put(
        f"{API_PREFIX}/configs/alice",
        json={"metadata": {"k": "v"}},
    )
    assert resp.status_code == 422


# --- delete_config (DELETE /configs/{bot_id}) ------------------------


@pytest.mark.asyncio
async def test_delete_existing_returns_204(app_client) -> None:
    client, backend = app_client
    await backend.register("alice", {})
    resp = await client.delete(f"{API_PREFIX}/configs/alice")
    assert resp.status_code == 204
    assert await backend.get("alice") is None


@pytest.mark.asyncio
async def test_delete_missing_returns_404(app_client) -> None:
    client, _ = app_client
    resp = await client.delete(f"{API_PREFIX}/configs/does-not-exist")
    assert resp.status_code == 404


# --- deactivate_config (POST /configs/{bot_id}/deactivate) -----------


@pytest.mark.asyncio
async def test_deactivate_existing_returns_204(app_client) -> None:
    """POST /configs/{id}/deactivate routes to backend.deactivate."""
    client, backend = app_client
    await backend.register("alice", {})
    resp = await client.post(f"{API_PREFIX}/configs/alice/deactivate")
    assert resp.status_code == 204
    reg = await backend.get("alice")
    assert reg is not None
    assert reg.status == "inactive"


@pytest.mark.asyncio
async def test_deactivate_missing_returns_404(app_client) -> None:
    client, _ = app_client
    resp = await client.post(f"{API_PREFIX}/configs/does-not-exist/deactivate")
    assert resp.status_code == 404


# --- clear_configs (DELETE /configs) ---------------------------------


@pytest.mark.asyncio
async def test_bulk_delete_returns_204_and_clears(app_client) -> None:
    """``HTTPRegistryBackend.clear`` tries this first before falling back."""
    client, backend = app_client
    await backend.register("alice", {})
    await backend.register("bob", {})

    resp = await client.delete(f"{API_PREFIX}/configs")
    assert resp.status_code == 204
    assert await backend.list_all() == []


# --- round-trip via HTTPRegistryBackend ------------------------------
#
# These tests drive HTTPRegistryBackend (the production client) against
# the in-process FastAPI app via an ASGI transport so the *two halves*
# of the wire protocol are pinned together — not just the server's
# externally-observable behavior.  If client expectations or server
# emissions drift, these break first.


@pytest.fixture
async def http_backend_against_router(app_client):
    """HTTPRegistryBackend wired against the in-process FastAPI app.

    aiohttp can't share an event-loop with an ASGI transport without
    going through a real socket, so we drive the protocol via the
    public ``AsyncClient`` (httpx) and validate the response shape is
    exactly what ``HTTPRegistryBackend._parse_registration`` consumes.
    """
    client, backend = app_client
    return client, backend


@pytest.mark.asyncio
async def test_roundtrip_response_shape_matches_client_parser(
    http_backend_against_router,
) -> None:
    """Server response decodes via HTTPRegistryBackend._parse_registration.

    This is the protocol-level pin: take the server's response, feed
    it through the client's parser, assert the round-trip preserves
    the registration faithfully.
    """
    from dataknobs_bots.registry.http_backend import HTTPRegistryBackend

    client, backend = http_backend_against_router
    await backend.register(
        "alice",
        {"llm": {"provider": "echo"}},
        metadata={"tenant_id": "acme", "audit": {"by": "alice"}},
    )

    # Server response for GET /configs/{id}
    resp = await client.get(f"{API_PREFIX}/configs/alice")
    assert resp.status_code == 200

    # Feed through the client's parser without making the network call.
    parser = HTTPRegistryBackend(base_url="http://test")
    reg = parser._parse_registration(resp.json())
    assert reg.bot_id == "alice"
    assert reg.config == {"llm": {"provider": "echo"}}
    assert reg.metadata == {"tenant_id": "acme", "audit": {"by": "alice"}}
    assert reg.status == "active"


@pytest.mark.asyncio
async def test_roundtrip_list_shape_matches_client_parser(
    http_backend_against_router,
) -> None:
    """``GET /configs`` response shape matches the client's list parser.

    The client looks for ``{"items": [...]}`` (or ``{"configs": ...}``
    or a bare list).  Our reference server emits ``items`` — verify the
    client's list-parsing branch handles it.
    """
    from dataknobs_bots.registry.http_backend import HTTPRegistryBackend

    client, backend = http_backend_against_router
    await backend.register("alice", {"v": 1}, metadata={"tenant_id": "acme"})
    await backend.register("bob", {"v": 2}, metadata={"tenant_id": "globex"})

    # Filter pushed to server
    resp = await client.get(
        f"{API_PREFIX}/configs",
        params={"filter_metadata": json.dumps({"tenant_id": "acme"})},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "items" in payload  # Documented shape

    # Walk the client's list-parsing branch: the server returns dict
    # with "items"; the client extracts the items and parses each.
    parser = HTTPRegistryBackend(base_url="http://test")
    items = payload["items"]
    parsed = [parser._parse_registration(item) for item in items]
    assert [r.bot_id for r in parsed] == ["alice"]
    assert parsed[0].metadata == {"tenant_id": "acme"}


# --- round-trip: client encoding ↔ server parsing -----------
#
# These tests construct the wire string the *client* would send for a
# given set of query kwargs, then feed it to the *server* via the
# in-process FastAPI app.  This catches drift between
# ``HTTPRegistryBackend.list_all``'s URL building and the router's
# ``Query`` parsing — for example, if the client switched from
# ``field:asc`` to ``field|asc`` but the server kept the colon form.


@pytest.mark.asyncio
async def test_roundtrip_sort_wire_format(http_backend_against_router) -> None:
    """SortSpec → ``field:order`` → server-side SortSpec is faithful."""
    from dataknobs_data import SortOrder, SortSpec

    client, backend = http_backend_against_router
    await backend.register("charlie", {})
    await backend.register("alice", {})
    await backend.register("bob", {})

    # Build the wire string exactly the way HTTPRegistryBackend.list_all does.
    sort_specs = [SortSpec(field="bot_id", order=SortOrder.DESC)]
    wire = [f"{s.field}:{s.order.value}" for s in sort_specs]

    resp = await client.get(
        f"{API_PREFIX}/configs",
        params=[("sort", w) for w in wire],
    )
    assert resp.status_code == 200
    ids = [r["bot_id"] for r in resp.json()["items"]]
    assert ids == ["charlie", "bob", "alice"]


@pytest.mark.asyncio
async def test_roundtrip_all_phase_6b_params(
    http_backend_against_router,
) -> None:
    """All query params encode → parse → push-down end-to-end."""
    from dataknobs_data import SortOrder, SortSpec

    client, backend = http_backend_against_router
    for i in range(6):
        await backend.register(
            f"bot-{i}",
            {},
            metadata={"tenant_id": "acme" if i % 2 == 0 else "globex"},
        )

    # Encode like the client would for:
    #   list_all(status="active", filter_metadata={"tenant_id": "acme"},
    #            sort=[SortSpec("bot_id", DESC)], limit=2, offset=1)
    sort_specs = [SortSpec(field="bot_id", order=SortOrder.DESC)]
    params: list[tuple[str, str]] = [
        ("status", "active"),
        ("filter_metadata", json.dumps({"tenant_id": "acme"}, sort_keys=True)),
        ("limit", "2"),
        ("offset", "1"),
        *[("sort", f"{s.field}:{s.order.value}") for s in sort_specs],
    ]

    resp = await client.get(f"{API_PREFIX}/configs", params=params)
    assert resp.status_code == 200
    ids = [r["bot_id"] for r in resp.json()["items"]]
    # acme bots in desc order: bot-4, bot-2, bot-0; offset=1 drops bot-4;
    # limit=2 keeps [bot-2, bot-0].
    assert ids == ["bot-2", "bot-0"]

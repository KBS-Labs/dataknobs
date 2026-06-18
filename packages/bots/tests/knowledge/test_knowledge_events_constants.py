"""Tests for the knowledge-layer event topic constants + helpers.

Covers the ``Final[str]`` topic-constant contract (a rename without a
matching changelog entry breaks the drift guard), the
``KnowledgeTriggerPayload`` runtime round-trip, and the
``TenantFilteredCallback`` short-circuit behaviour.
"""

from __future__ import annotations

import json

from dataknobs_bots.knowledge import (
    INGEST_DOMAIN_END,
    INGEST_DOMAIN_START,
    INGEST_METADATA_WRITE,
    INGEST_SNAPSHOT_WRITE,
    KnowledgeTriggerPayload,
    TenantFilteredCallback,
)


def test_topic_constants_stable() -> None:
    """Drift guard: literal topic values are part of the cross-package
    contract. A rename must be a deliberate, changelogged change."""
    assert INGEST_DOMAIN_START == "ingest:domain:start"
    assert INGEST_DOMAIN_END == "ingest:domain:end"
    assert INGEST_METADATA_WRITE == "ingest:metadata:write"
    assert INGEST_SNAPSHOT_WRITE == "ingest:snapshot:write"


def test_trigger_payload_json_roundtrip() -> None:
    """The wire format is ``dict[str, Any]``; the TypedDict is just a
    documented shape. A constructed payload round-trips through JSON."""
    payload: KnowledgeTriggerPayload = {
        "domain_id": "d1",
        "tenant_id": "acme",
        "key": "d1/content/a.md",
        "since_version": "v1",
        "force_full": False,
        "last_version": "v0",
    }
    restored = json.loads(json.dumps(payload))
    assert restored == payload
    # Minimal (only the required field) is also valid.
    minimal: KnowledgeTriggerPayload = {"domain_id": "d1"}
    assert json.loads(json.dumps(minimal)) == {"domain_id": "d1"}


def test_tenant_filtered_callback_matches_only_bound_tenant() -> None:
    hits: list[dict] = []
    cb = TenantFilteredCallback(hits.append, tenant_id="acme")

    cb({"tenant_id": "acme", "domain_id": "d1"})
    cb({"tenant_id": "other", "domain_id": "d1"})
    cb({"domain_id": "d1"})  # no tenant_id key at all

    assert len(hits) == 1
    assert hits[0]["tenant_id"] == "acme"


def test_tenant_filtered_callback_returns_inner_result_on_match() -> None:
    cb = TenantFilteredCallback(lambda ev: ev["domain_id"], tenant_id="acme")
    assert cb({"tenant_id": "acme", "domain_id": "d1"}) == "d1"
    assert cb({"tenant_id": "nope", "domain_id": "d1"}) is None


def test_tenant_filtered_callback_repr() -> None:
    cb = TenantFilteredCallback(lambda ev: None, tenant_id="acme")
    assert "tenant_id='acme'" in repr(cb)

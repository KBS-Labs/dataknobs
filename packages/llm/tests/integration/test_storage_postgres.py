"""Postgres integration tests for DataknobsConversationStorage filter_metadata.

Memory-backend coverage masks the metadata-column propagation gap because
``Record.get_nested_value("metadata.X")`` falls back to the data column on
the memory backend's ``Filter.matches`` flow.  Postgres routes
``metadata.X`` filters to the dedicated metadata column directly, so these
tests exercise the path that originally surfaced the bug (admin endpoint
returning zero rows on AWS RDS Postgres even though ``state.metadata`` was
populated).

Gated by ``@requires_postgres`` and ``TEST_POSTGRES=true`` (matches the
``packages/data/tests/integration`` convention; ``bin/test.sh`` exports
``TEST_POSTGRES=true`` by default).
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest
from dataknobs_common.testing import requires_postgres, safe_sql_ident
from dataknobs_structures.tree import Tree

from dataknobs_llm.conversations import (
    ConversationNode,
    ConversationState,
    DataknobsConversationStorage,
)
from dataknobs_llm.llm.base import LLMMessage

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("TEST_POSTGRES", "").lower() != "true",
        reason="Postgres integration tests require TEST_POSTGRES=true and a "
        "running postgres instance (bin/dk up).",
    ),
    requires_postgres,
]


def _make_state(
    conversation_id: str, metadata: dict[str, str]
) -> ConversationState:
    root = ConversationNode(
        message=LLMMessage(role="system", content="x"),
        node_id="",
    )
    return ConversationState(
        conversation_id=conversation_id,
        message_tree=Tree(root),
        metadata=metadata,
    )


def _backend_config(postgres_test_db: dict[str, Any]) -> dict[str, Any]:
    """Build an AsyncPostgresDatabase config dict from the fixture."""
    return {
        "host": postgres_test_db["host"],
        "port": postgres_test_db["port"],
        "user": postgres_test_db["user"],
        "password": postgres_test_db["password"],
        "database": postgres_test_db["database"],
        "schema": postgres_test_db["schema"],
        "table": postgres_test_db["table"],
    }


async def test_list_conversations_filter_by_metadata_postgres(
    postgres_test_db: dict[str, Any],
) -> None:
    """``filter_metadata`` must hit the metadata column on Postgres.

    Reproduces the admin-endpoint bug: with the producer populating
    ``state.metadata`` correctly, ``list_conversations`` must return
    matching rows. Pre-fix this returned [] because ``record.metadata``
    was empty and the SQL filter routed to a NULL column.
    """
    from dataknobs_data.backends.postgres import AsyncPostgresDatabase

    backend = AsyncPostgresDatabase(_backend_config(postgres_test_db))
    await backend.connect()
    try:
        storage = DataknobsConversationStorage(backend)

        for user in ["alice", "alice", "bob"]:
            await storage.save_conversation(
                _make_state(
                    conversation_id=f"conv-{uuid.uuid4().hex[:12]}",
                    metadata={
                        "user_id": user,
                        "domain_id": "prompt-engineering",
                    },
                )
            )

        alice_convs = await storage.list_conversations(
            filter_metadata={"user_id": "alice"}
        )
        assert len(alice_convs) == 2
        assert all(c.metadata["user_id"] == "alice" for c in alice_convs)

        # Multi-key filter (the admin-listing shape):
        domain_convs = await storage.list_conversations(
            filter_metadata={"domain_id": "prompt-engineering"}
        )
        assert len(domain_convs) == 3

        # No-match: regression guard against false positives.
        empty = await storage.list_conversations(
            filter_metadata={"user_id": "nobody"}
        )
        assert empty == []
    finally:
        await backend.close()


async def test_count_conversations_filter_by_metadata_postgres(
    postgres_test_db: dict[str, Any],
) -> None:
    """``count_conversations`` follows the same filter path as
    ``list_conversations``; cover it explicitly so the route doesn't
    silently regress.
    """
    from dataknobs_data.backends.postgres import AsyncPostgresDatabase

    backend = AsyncPostgresDatabase(_backend_config(postgres_test_db))
    await backend.connect()
    try:
        storage = DataknobsConversationStorage(backend)

        for user in ["alice", "alice", "bob"]:
            await storage.save_conversation(
                _make_state(
                    conversation_id=f"conv-{uuid.uuid4().hex[:12]}",
                    metadata={"user_id": user},
                )
            )

        assert (
            await storage.count_conversations(
                filter_metadata={"user_id": "alice"}
            )
            == 2
        )
        assert (
            await storage.count_conversations(
                filter_metadata={"user_id": "bob"}
            )
            == 1
        )
        assert (
            await storage.count_conversations(
                filter_metadata={"user_id": "nobody"}
            )
            == 0
        )
    finally:
        await backend.close()


async def test_metadata_column_populated_on_postgres(
    postgres_test_db: dict[str, Any],
) -> None:
    """Lower-level guard: assert the row's metadata column is non-NULL
    after a save.

    Catches the producer bug regardless of whether the SQL filter routing
    is also broken — narrowing diagnosis when a future regression
    surfaces.
    """
    import asyncpg

    from dataknobs_data.backends.postgres import AsyncPostgresDatabase

    backend = AsyncPostgresDatabase(_backend_config(postgres_test_db))
    await backend.connect()
    try:
        storage = DataknobsConversationStorage(backend)
        await storage.save_conversation(
            _make_state(
                conversation_id="conv-md-col-test",
                metadata={"domain_id": "prompt-engineering"},
            )
        )

        # Direct asyncpg query — independent of any dataknobs filter
        # routing — to confirm the column is populated.
        conn = await asyncpg.connect(
            host=postgres_test_db["host"],
            port=postgres_test_db["port"],
            user=postgres_test_db["user"],
            password=postgres_test_db["password"],
            database=postgres_test_db["database"],
        )
        try:
            row = await conn.fetchrow(
                f"SELECT metadata FROM {safe_sql_ident(postgres_test_db['schema'])}."
                f"{safe_sql_ident(postgres_test_db['table'])} WHERE id = $1",
                "conv-md-col-test",
            )
            assert row is not None
            assert row["metadata"] is not None
            md_raw = row["metadata"]
            if isinstance(md_raw, str):
                import json

                md = json.loads(md_raw)
            else:
                md = md_raw
            assert md.get("domain_id") == "prompt-engineering"
        finally:
            await conn.close()
    finally:
        await backend.close()

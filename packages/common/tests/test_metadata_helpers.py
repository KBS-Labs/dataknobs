"""Unit tests for ``dataknobs_common.metadata.enforce_immutable_keys``.

These tests pin the contract of the layered-merge primitive used by
``VectorMemory`` (tenant-scope enforcement), ``RAGKnowledgeBase``
(chunk-text protection), and the markdown chunker (node-classification
protection).
"""

from __future__ import annotations

import logging

import pytest

from dataknobs_common.metadata import enforce_immutable_keys


class TestEnforceImmutableKeys:
    """Pin the layered-merge primitive's contract."""

    def test_empty_keys_is_noop(self) -> None:
        """With no immutable keys, target is returned unchanged."""
        target = {"user_id": "ATTACKER", "content": "hello"}
        source = {"user_id": "tenant-A"}
        result = enforce_immutable_keys(
            target=target,
            caller={"user_id": "ATTACKER"},
            source=source,
            keys=set(),
        )
        # Target reflects pre-call state (caller already merged).
        assert result["user_id"] == "ATTACKER"

    def test_immutable_key_blocks_caller_override(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Immutable key takes value from source, regardless of caller."""
        target = {"user_id": "ATTACKER", "category": "support"}
        source = {"user_id": "tenant-A"}
        with caplog.at_level(logging.WARNING):
            result = enforce_immutable_keys(
                target=target,
                caller={"user_id": "ATTACKER", "category": "support"},
                source=source,
                keys={"user_id"},
            )
        assert result["user_id"] == "tenant-A"
        # Non-immutable keys preserved.
        assert result["category"] == "support"
        # Warning emitted naming the key.
        assert any(
            "user_id" in record.message and "immutable" in record.message.lower()
            for record in caplog.records
        )

    def test_silent_when_caller_agrees(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning when caller-supplied value matches the source value."""
        target = {"user_id": "tenant-A"}
        source = {"user_id": "tenant-A"}
        with caplog.at_level(logging.WARNING):
            result = enforce_immutable_keys(
                target=target,
                caller={"user_id": "tenant-A"},
                source=source,
                keys={"user_id"},
            )
        assert result["user_id"] == "tenant-A"
        # No warning records — caller didn't try to override.
        assert not any(
            "immutable" in record.message.lower() for record in caplog.records
        )

    def test_immutable_key_absent_from_source_skipped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Keys named immutable but absent from source are not enforced."""
        target = {"user_id": "ATTACKER"}
        source: dict = {}  # no user_id in source
        with caplog.at_level(logging.WARNING):
            result = enforce_immutable_keys(
                target=target,
                caller={"user_id": "ATTACKER"},
                source=source,
                keys={"user_id"},
            )
        # Source has no opinion, so caller value passes through.
        assert result["user_id"] == "ATTACKER"
        # No warning — source provided no value to override against.
        assert not any(
            "immutable" in record.message.lower() for record in caplog.records
        )

    def test_caller_none_is_safe(self) -> None:
        """``caller=None`` is treated as no caller-supplied keys."""
        target = {"user_id": "tenant-A"}
        source = {"user_id": "tenant-A"}
        result = enforce_immutable_keys(
            target=target,
            caller=None,
            source=source,
            keys={"user_id"},
        )
        assert result["user_id"] == "tenant-A"

    def test_array_valued_metadata_does_not_crash(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Array-valued metadata (numpy or list) does not raise on equality.

        Pre-fix, the helper used ``caller[key] != source_value`` which
        raises ``ValueError: ambiguous truth value`` when both values
        are numpy arrays — the comparison returns an element-wise
        array, and the implicit ``bool`` conversion fails. Vector
        metadata is the realistic vector for hitting this path.
        """
        np = pytest.importorskip("numpy")

        embedding = np.array([0.1, 0.2, 0.3])
        target = {"embedding": embedding, "user_id": "tenant-A"}
        source = {"embedding": embedding}
        with caplog.at_level(logging.WARNING):
            # MUST not raise.
            result = enforce_immutable_keys(
                target=target,
                caller={"embedding": embedding},
                source=source,
                keys={"embedding"},
            )
        # Source value preserved, no spurious warning when arrays agree.
        assert result["embedding"] is embedding

    def test_array_valued_caller_override_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Array-valued caller override warns and discards caller value."""
        np = pytest.importorskip("numpy")

        source_embedding = np.array([0.1, 0.2, 0.3])
        caller_embedding = np.array([0.9, 0.9, 0.9])
        target = {"embedding": caller_embedding}
        source = {"embedding": source_embedding}
        with caplog.at_level(logging.WARNING):
            result = enforce_immutable_keys(
                target=target,
                caller={"embedding": caller_embedding},
                source=source,
                keys={"embedding"},
            )
        # Source value wins.
        assert np.array_equal(result["embedding"], source_embedding)
        # Warning emitted.
        assert any(
            "embedding" in record.message and "immutable" in record.message.lower()
            for record in caplog.records
        )

    def test_list_valued_metadata_does_not_crash(self) -> None:
        """List-valued metadata (without numpy) compares safely."""
        target = {"tags": ["a", "b"], "user_id": "tenant-A"}
        source = {"tags": ["a", "b"]}
        result = enforce_immutable_keys(
            target=target,
            caller={"tags": ["a", "b"]},
            source=source,
            keys={"tags"},
        )
        assert result["tags"] == ["a", "b"]

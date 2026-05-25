"""Tests for ``RAGKnowledgeBaseConfig`` polymorphic-section validation.

The bots-side adopter of polymorphic-section validation: a parsed
``RAGKnowledgeBaseConfig`` can be validated (without constructing the
knowledge base) so a malformed or unknown ``vector_store`` section is caught
at config-lint time. These config-level tests construct the config directly
— a legitimate use (they test config internals, not bot flows), so no
``BotTestHarness`` is needed.

``dataknobs_data.vector.stores`` is imported so its ``vector_store``
resolver is registered (eager on import); ``dataknobs-bots`` depends on
``dataknobs-data`` so this is always available.
"""

from __future__ import annotations

import dataknobs_data.vector.stores  # noqa: F401 — eager resolver registration
import pytest

from dataknobs_bots.knowledge.config import RAGKnowledgeBaseConfig

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.testing import assert_polymorphic_bindings_resolve


def test_good_vector_store_validates() -> None:
    cfg = RAGKnowledgeBaseConfig.from_dict(
        {"vector_store": {"backend": "memory", "dimensions": 384}}
    )
    cfg.validate()


def test_default_empty_vector_store_is_noop() -> None:
    # vector_store defaults to {} (the from_components path) — skipped.
    RAGKnowledgeBaseConfig.from_dict({}).validate()


def test_unknown_backend_raises() -> None:
    cfg = RAGKnowledgeBaseConfig.from_dict({"vector_store": {"backend": "pgvektor"}})
    with pytest.raises(ConfigurationError) as exc:
        cfg.validate()
    msg = str(exc.value)
    assert "vector_store" in msg
    assert "RAGKnowledgeBaseConfig" in msg


def test_bad_nested_field_raises() -> None:
    # A bad nested timestamps.format is a field-level error surfaced by the
    # dry-run build of the resolved vector-store config.
    cfg = RAGKnowledgeBaseConfig.from_dict(
        {"vector_store": {"backend": "memory", "timestamps": {"format": "bogus"}}}
    )
    with pytest.raises(ValueError, match="timestamps.format"):
        cfg.validate()


def test_bindings_resolve_guard() -> None:
    assert_polymorphic_bindings_resolve(RAGKnowledgeBaseConfig)

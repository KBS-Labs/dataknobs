"""Tests for ``RAGKnowledgeBaseConfig`` polymorphic-section validation.

The bots-side adopter of polymorphic-section validation: a parsed
``RAGKnowledgeBaseConfig`` can be validated (without constructing the
knowledge base) so a malformed or unknown ``vector_store`` or ``embedding``
section is caught at config-lint time. These config-level tests construct the
config directly — a legitimate use (they test config internals, not bot
flows), so no ``BotTestHarness`` is needed.

``dataknobs_data.vector.stores`` is imported so its ``vector_store``
resolver is registered (eager on import) and ``dataknobs_llm`` so its
``embedding`` resolver is; ``dataknobs-bots`` depends on both so they are
always available.
"""

from __future__ import annotations

# Required side-effect imports: importing these packages registers the
# "vector_store" and "embedding" resolvers in config_registries, which
# validate() resolves those sections against. Do NOT remove as "unused" —
# without them the bindings are unregistered and validate() degrades to a
# no-op skip.
import dataknobs_data.vector.stores  # noqa: F401
import dataknobs_llm  # noqa: F401
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


# --- nested embedding section (reuses the LLMConfig resolver) ---


def test_good_embedding_validates() -> None:
    cfg = RAGKnowledgeBaseConfig.from_dict(
        {"embedding": {"provider": "echo", "model": "test-embed"}}
    )
    cfg.validate()


def test_default_absent_embedding_is_noop() -> None:
    # embedding defaults to None — skipped (empty-section rule).
    RAGKnowledgeBaseConfig.from_dict({}).validate()


def test_unknown_embedding_provider_raises() -> None:
    cfg = RAGKnowledgeBaseConfig.from_dict(
        {"embedding": {"provider": "no-such-provider", "model": "x"}}
    )
    with pytest.raises(ConfigurationError) as exc:
        cfg.validate()
    msg = str(exc.value)
    assert "embedding" in msg
    assert "RAGKnowledgeBaseConfig" in msg


def test_flat_embedding_keys_not_validated() -> None:
    # Legacy flat keys (no nested ``embedding`` dict) are intentionally not
    # validated — the nested section is empty, so validate() skips it even
    # though the flat provider is bogus. Documents the flat-key decision.
    RAGKnowledgeBaseConfig.from_dict(
        {"embedding_provider": "no-such-provider", "embedding_model": "x"}
    ).validate()


# --- both bindings (vector_store + embedding) resolve ---


def test_bindings_resolve_guard() -> None:
    # Now covers both the vector_store and embedding bindings.
    assert_polymorphic_bindings_resolve(RAGKnowledgeBaseConfig)

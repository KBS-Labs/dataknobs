"""Tests for ``DynaBotConfig`` / ``CompositeMemoryConfig`` /
``SummaryMemoryConfig`` polymorphic-section validation.

The bots-side adopters of polymorphic-section validation: a parsed
``DynaBotConfig`` can be validated (without constructing the bot) so a
malformed or unknown ``llm`` / ``memory`` / ``knowledge_base`` section — and
the nested ``vector_store`` reached through the knowledge base — is caught at
config-lint time. ``CompositeMemoryConfig`` validates each ``strategies``
element, and ``SummaryMemoryConfig`` its ``llm`` section, by the same
mechanism.

These config-level tests construct the config dataclass directly — a
legitimate use (they test config internals, not bot flows), so no
``BotTestHarness`` is needed.

The registry modules are imported for their side effect (registering the
``memory`` / ``knowledge_base`` resolvers, eager on import); ``dataknobs-data``
is imported so the nested ``vector_store`` resolver is registered too, and
``dataknobs_llm`` so the ``llm`` resolver is registered.
"""

from __future__ import annotations

from collections.abc import Callable

# Required side-effect imports: registering the resolvers in config_registries
# that validate() resolves the sections against. Do NOT remove as "unused" —
# without them the bindings are unregistered and validate() degrades to a
# no-op skip.
import dataknobs_bots.knowledge.registry  # noqa: F401
import dataknobs_bots.memory.registry  # noqa: F401
import dataknobs_data.vector.stores  # noqa: F401
import dataknobs_llm  # noqa: F401
import pytest

from dataknobs_bots.bot.config import DynaBotConfig
from dataknobs_bots.memory.config import CompositeMemoryConfig, SummaryMemoryConfig
from dataknobs_bots.memory.registry import memory_backends

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.testing import assert_polymorphic_bindings_resolve


# --- bindings resolve (wiring guard) ---


def test_dynabot_bindings_resolve_guard() -> None:
    assert_polymorphic_bindings_resolve(DynaBotConfig)


def test_composite_bindings_resolve_guard() -> None:
    assert_polymorphic_bindings_resolve(CompositeMemoryConfig)


def test_summary_memory_bindings_resolve_guard() -> None:
    # SummaryMemoryConfig declares the ``llm`` binding; this fails in CI if the
    # dataknobs-llm resolver registration ever drops or is renamed.
    assert_polymorphic_bindings_resolve(SummaryMemoryConfig)


# --- happy paths ---


def test_good_memory_section_validates() -> None:
    DynaBotConfig.from_dict({"memory": {"type": "buffer", "max_messages": 25}}).validate()


def test_good_llm_section_validates() -> None:
    DynaBotConfig.from_dict(
        {"llm": {"provider": "ollama", "model": "llama3.2", "temperature": 0.7}}
    ).validate()


def test_good_knowledge_base_section_validates() -> None:
    DynaBotConfig.from_dict(
        {
            "knowledge_base": {
                "type": "rag",
                "vector_store": {"backend": "memory", "dimensions": 384},
            }
        }
    ).validate()


def test_empty_sections_are_noop() -> None:
    # memory/knowledge_base default to None; an empty config is a clean no-op.
    DynaBotConfig.from_dict({}).validate()
    DynaBotConfig.from_dict({"memory": None, "knowledge_base": None}).validate()


# --- unknown discriminator -> ConfigurationError ---


def test_unknown_memory_type_raises() -> None:
    cfg = DynaBotConfig.from_dict({"memory": {"type": "bufferr"}})
    with pytest.raises(ConfigurationError) as exc:
        cfg.validate()
    msg = str(exc.value)
    assert "memory" in msg
    assert "DynaBotConfig" in msg


def test_unknown_knowledge_base_type_raises() -> None:
    cfg = DynaBotConfig.from_dict({"knowledge_base": {"type": "raag"}})
    with pytest.raises(ConfigurationError) as exc:
        cfg.validate()
    msg = str(exc.value)
    assert "knowledge_base" in msg
    assert "DynaBotConfig" in msg


def test_unknown_llm_provider_raises() -> None:
    cfg = DynaBotConfig.from_dict({"llm": {"provider": "bogus-provider", "model": "x"}})
    with pytest.raises(ConfigurationError) as exc:
        cfg.validate()
    msg = str(exc.value)
    assert "llm" in msg
    assert "DynaBotConfig" in msg


# --- SummaryMemoryConfig.llm validated by the same mechanism ---


def test_summary_memory_good_llm_validates() -> None:
    SummaryMemoryConfig.from_dict(
        {"recent_window": 5, "llm": {"provider": "echo", "model": "test"}}
    ).validate()


def test_summary_memory_absent_llm_is_noop() -> None:
    # llm defaults to None on SummaryMemoryConfig — a clean no-op.
    SummaryMemoryConfig.from_dict({"recent_window": 5}).validate()


def test_summary_memory_unknown_llm_provider_raises() -> None:
    cfg = SummaryMemoryConfig.from_dict({"llm": {"provider": "nope"}})
    with pytest.raises(ConfigurationError, match="llm"):
        cfg.validate()


# --- recursion through knowledge_base -> vector_store ---


def test_bad_nested_vector_store_via_kb_recursion_raises() -> None:
    # A single DynaBotConfig.validate() descends: knowledge_base resolves to
    # RAGKnowledgeBaseConfig, which carries the vector_store binding, so a bad
    # nested timestamps.format surfaces from the dry-run vector-store build.
    cfg = DynaBotConfig.from_dict(
        {
            "knowledge_base": {
                "type": "rag",
                "vector_store": {
                    "backend": "memory",
                    "timestamps": {"format": "bogus"},
                },
            }
        }
    )
    with pytest.raises(ValueError, match="timestamps.format"):
        cfg.validate()


def test_unknown_nested_vector_store_backend_via_kb_recursion_raises() -> None:
    cfg = DynaBotConfig.from_dict(
        {"knowledge_base": {"type": "rag", "vector_store": {"backend": "pgvektor"}}}
    )
    with pytest.raises(ConfigurationError, match="vector_store"):
        cfg.validate()


# --- composite strategies validated element-wise ---


def test_composite_strategy_element_validated() -> None:
    # Directly on CompositeMemoryConfig: a bad element type surfaces.
    cfg = CompositeMemoryConfig.from_dict(
        {"strategies": [{"type": "buffer"}, {"type": "bogus_strategy"}]}
    )
    with pytest.raises(ConfigurationError, match="strategies"):
        cfg.validate()


def test_composite_strategy_element_validated_via_dynabot() -> None:
    # And through a top-level DynaBotConfig whose memory is a composite: the
    # memory resolver yields CompositeMemoryConfig, whose own validate()
    # checks each strategy element.
    cfg = DynaBotConfig.from_dict(
        {
            "memory": {
                "type": "composite",
                "strategies": [{"type": "buffer"}, {"type": "bogus_strategy"}],
            }
        }
    )
    with pytest.raises(ConfigurationError, match="strategies"):
        cfg.validate()


def test_composite_all_good_strategies_validates() -> None:
    CompositeMemoryConfig.from_dict(
        {
            "strategies": [
                {"type": "buffer", "max_messages": 10},
                {"type": "summary", "recent_window": 5},
            ]
        }
    ).validate()


# --- skip sentinel: bare-callable backend skipped, not raised ---


def test_bare_callable_backend_skipped_without_raising(
    register_untyped_backend: Callable[..., str],
) -> None:
    # A custom backend registered as a bare callable has no CONFIG_CLS; the
    # resolver returns SKIP_VALIDATION, so validate() skips its section rather
    # than false-positive-raising on a valid, constructible backend.
    register_untyped_backend(memory_backends)
    DynaBotConfig.from_dict(
        {"memory": {"type": "untyped_test_backend", "anything": 1}}
    ).validate()

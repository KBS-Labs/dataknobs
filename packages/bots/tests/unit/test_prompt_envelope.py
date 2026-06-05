"""Tests for the prompt envelope helper and its integration into DynaBot.

Covers:

- ``_build_message_with_context`` produces a non-XML markdown envelope
  by default — the bug this work is for: XML-wrapped user prompts cue
  small instruction-tuned models to complete with mirroring
  ``<response>...</response>`` wrappers, which persist in conversation
  history and self-reinforce.
- The ``PromptEnvelope`` helper itself renders the three styles
  ``markdown`` / ``xml`` / ``prose`` to fixed shapes (golden-string
  parity).
- ``DynaBotConfig.prompt_envelope = "xml"`` rounds back to the legacy
  byte shape so consumers needing the old behavior have a one-line
  escape hatch.
- ``DynaBotConfig.prompt_envelope`` validates its string value.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from dataknobs_bots.bot.base import DynaBot
from dataknobs_bots.bot.config import DynaBotConfig
from dataknobs_bots.prompts import PromptEnvelope, PromptEnvelopeStyle

TEST_DOCS = Path(__file__).parent.parent / "test_docs"

# A regex broad enough to catch any XML element name that the bot
# could emit through _build_message_with_context — we want zero hits
# in the default markdown envelope.
_ANY_XML_TAG = re.compile(r"<[a-z_][a-z0-9_]*>")


def _kb_config() -> dict[str, object]:
    return {
        "llm": {"provider": "echo", "model": "test"},
        "conversation_storage": {"backend": "memory"},
        "knowledge_base": {
            "enabled": True,
            "type": "rag",
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
            "documents_path": str(TEST_DOCS),
        },
    }


# ---------------------------------------------------------------------------
# E1 — reproducing pin: the bot's default auto-context user prompt is markdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_message_with_context_default_is_markdown_no_xml() -> None:
    """Default `_build_message_with_context` output contains no XML tags.

    Pre-fix HEAD produces ``<conversation_history>...``, ``<question>...``,
    and (when KB returns hits) ``<knowledge_base>...``. After the fix,
    the default envelope is markdown — no XML tags appear and the
    expected ``## Label`` headings + ``---`` separator are present.
    """
    bot = await DynaBot.from_config(_kb_config())

    message = await bot._build_message_with_context("How do I configure memory?")

    # No XML envelope tags of any kind.
    assert _ANY_XML_TAG.search(message) is None, (
        "Default envelope must not produce XML tags; got: %r" % message
    )
    # The KB query against the bundled test_docs returns hits, so we
    # expect Knowledge base + Question sections joined by the markdown
    # rule. (Memory is absent in this config.)
    assert "## Knowledge base" in message
    assert "## Question" in message
    assert "\n\n---\n\n" in message
    assert message.rstrip().endswith("How do I configure memory?")


# ---------------------------------------------------------------------------
# E2 — golden-string parity for the three styles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("style", "expected_kb", "expected_history", "expected_question", "joiner"),
    [
        (
            PromptEnvelopeStyle.MARKDOWN,
            "## Knowledge base\n\nKB body",
            "## Conversation history\n\nHistory body",
            "## Question\n\nWhat is X?",
            "\n\n---\n\n",
        ),
        (
            PromptEnvelopeStyle.XML,
            "<knowledge_base>\nKB body\n</knowledge_base>",
            "<conversation_history>\nHistory body\n</conversation_history>",
            "<question>\nWhat is X?\n</question>",
            "\n\n",
        ),
        (
            PromptEnvelopeStyle.PROSE,
            "Knowledge base:\n\nKB body",
            "Conversation history:\n\nHistory body",
            "Question:\n\nWhat is X?",
            "\n\n",
        ),
    ],
)
def test_envelope_style_golden_strings(
    style: PromptEnvelopeStyle,
    expected_kb: str,
    expected_history: str,
    expected_question: str,
    joiner: str,
) -> None:
    """Each style renders the three labeled sections to a fixed shape."""
    env = PromptEnvelope(style)

    assert env.section("Knowledge base", "KB body", tag="knowledge_base") == expected_kb
    assert (
        env.section(
            "Conversation history", "History body", tag="conversation_history"
        )
        == expected_history
    )
    assert env.section("Question", "What is X?", tag="question") == expected_question
    assert env.joiner() == joiner


def test_envelope_empty_body_returns_empty_string() -> None:
    """An empty body yields ``""`` regardless of style — no stray header."""
    for style in PromptEnvelopeStyle:
        env = PromptEnvelope(style)
        assert env.section("Knowledge base", "", tag="knowledge_base") == ""


# ---------------------------------------------------------------------------
# E3 — back-compat escape hatch: `prompt_envelope: "xml"` reproduces HEAD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_xml_envelope_reproduces_legacy_byte_shape() -> None:
    """``prompt_envelope: "xml"`` yields the legacy XML envelope shape.

    Consumers depending on the previous bytes can pin this in config
    and migrate at their own pace.
    """
    config = _kb_config()
    config["prompt_envelope"] = "xml"

    bot = await DynaBot.from_config(config)

    message = await bot._build_message_with_context("How do I configure memory?")

    assert "<knowledge_base>\n" in message
    assert "\n</knowledge_base>" in message
    assert "<question>\nHow do I configure memory?\n</question>" in message
    # The markdown-only markers must NOT appear when style is xml.
    # NB: KB-formatted content can itself contain ``\n\n---\n\n``
    # (``ContextFormatter`` uses it as a group separator inside the KB
    # body); the joiner we want to rule out is the *envelope* joiner
    # between sections — the transition from ``</knowledge_base>`` to
    # ``<question>``.
    assert "## Knowledge base" not in message
    assert "</knowledge_base>\n\n<question>" in message
    assert "</knowledge_base>\n\n---\n\n" not in message


# ---------------------------------------------------------------------------
# E4 — DynaBotConfig validation
# ---------------------------------------------------------------------------


def test_dynabotconfig_rejects_invalid_prompt_envelope() -> None:
    """An unknown ``prompt_envelope`` value raises ``ValueError`` with a clear message."""
    with pytest.raises(ValueError, match="prompt_envelope must be one of"):
        DynaBotConfig(prompt_envelope="invalid")


def test_dynabotconfig_default_prompt_envelope_is_markdown() -> None:
    """The default ``prompt_envelope`` is ``"markdown"`` — the bug fix's chosen default."""
    assert DynaBotConfig().prompt_envelope == "markdown"


def test_dynabotconfig_prompt_envelope_round_trips() -> None:
    """``prompt_envelope`` round-trips through ``from_dict`` / ``to_dict``."""
    cfg = DynaBotConfig.from_dict({"prompt_envelope": "xml"})
    assert cfg.prompt_envelope == "xml"
    assert DynaBotConfig.from_dict(cfg.to_dict()) == cfg

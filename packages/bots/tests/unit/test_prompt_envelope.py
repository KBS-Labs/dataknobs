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

    # Pre-check: the test is meaningful only when the KB actually
    # returns hits — otherwise no sections are built, the bare message
    # is returned, and "no XML tags" passes vacuously without
    # exercising the envelope. The echo embedder is deterministic, so
    # this should always hold, but the guard documents the dependency
    # and fails loudly if the bundled test_docs / embedder behavior
    # ever changes.
    assert bot.knowledge_base is not None
    kb_results = await bot.knowledge_base.query(
        "How do I configure memory?", k=5,
    )
    assert kb_results, (
        "test_docs must yield at least one hit so this test exercises "
        "the envelope rendering rather than the bare-message bypass"
    )

    message = await bot._build_message_with_context("How do I configure memory?")

    # No XML envelope tags of any kind.
    assert _ANY_XML_TAG.search(message) is None, (
        "Default envelope must not produce XML tags; got: %r" % message
    )
    # The KB query returned hits above, so Knowledge base + Question
    # sections joined by the markdown rule must be present. (Memory is
    # absent in this config.)
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
# Named section helpers — one source of truth for the (label, tag) pairs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("style", list(PromptEnvelopeStyle))
def test_knowledge_base_section_matches_generic_section(style: PromptEnvelopeStyle) -> None:
    """``knowledge_base_section(body)`` produces the same bytes as the generic call."""
    env = PromptEnvelope(style)
    expected = env.section("Knowledge base", "KB body", tag="knowledge_base")
    assert env.knowledge_base_section("KB body") == expected


@pytest.mark.parametrize("style", list(PromptEnvelopeStyle))
def test_conversation_history_section_matches_generic_section(
    style: PromptEnvelopeStyle,
) -> None:
    """``conversation_history_section(body)`` matches the generic call."""
    env = PromptEnvelope(style)
    expected = env.section(
        "Conversation history", "History body", tag="conversation_history"
    )
    assert env.conversation_history_section("History body") == expected


@pytest.mark.parametrize("style", list(PromptEnvelopeStyle))
def test_question_section_matches_generic_section(style: PromptEnvelopeStyle) -> None:
    """``question_section(body)`` matches the generic call."""
    env = PromptEnvelope(style)
    expected = env.section("Question", "What is X?", tag="question")
    assert env.question_section("What is X?") == expected


def test_section_helpers_empty_body_returns_empty_string() -> None:
    """The named helpers honor the empty-body → ``""`` contract."""
    for style in PromptEnvelopeStyle:
        env = PromptEnvelope(style)
        assert env.knowledge_base_section("") == ""
        assert env.conversation_history_section("") == ""
        assert env.question_section("") == ""


def test_section_for_tag_canonical_label_resolution() -> None:
    """``section_for_tag`` pulls the canonical label from SECTION_LABELS."""
    from dataknobs_bots.prompts import SECTION_LABELS

    env = PromptEnvelope(PromptEnvelopeStyle.MARKDOWN)
    for tag, label in SECTION_LABELS.items():
        rendered = env.section_for_tag(tag, "body")
        assert rendered == f"## {label}\n\nbody"


def test_section_for_tag_unknown_tag_falls_back_to_derivation() -> None:
    """Unknown tags get the ``"foo_bar"`` → ``"Foo bar"`` derivation, not a crash."""
    env = PromptEnvelope(PromptEnvelopeStyle.MARKDOWN)
    rendered = env.section_for_tag("custom_context", "body")
    assert rendered == "## Custom context\n\nbody"


def test_section_labels_is_immutable() -> None:
    """The canonical mapping is exposed as a read-only view."""
    from dataknobs_bots.prompts import SECTION_LABELS

    with pytest.raises(TypeError):
        SECTION_LABELS["new"] = "New"  # type: ignore[index]


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


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("XML", "xml"),
        ("Markdown", "markdown"),
        ("PROSE", "prose"),
        ("xMl", "xml"),
    ],
)
def test_dynabotconfig_prompt_envelope_is_case_insensitive(
    raw: str, normalized: str,
) -> None:
    """Mixed-case ``prompt_envelope`` values are accepted and normalized.

    YAML configs are human-written; rejecting ``"XML"`` would surprise
    consumers. The snapshot stores the lowercase form so downstream
    enum lookups match.
    """
    cfg = DynaBotConfig(prompt_envelope=raw)
    assert cfg.prompt_envelope == normalized


def test_dynabotconfig_invalid_envelope_error_mentions_case_insensitive() -> None:
    """The validation error names all accepted styles and notes case-insensitivity."""
    with pytest.raises(ValueError, match="case-insensitive"):
        DynaBotConfig(prompt_envelope="json")


# ---------------------------------------------------------------------------
# E5 — explicit byte-identity test: XML envelope == pre-fix bytes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_xml_envelope_byte_identical_to_legacy_assembly() -> None:
    """The XML envelope produces a byte-for-byte match against a hand-built legacy assembly.

    The PR claims ``prompt_envelope: "xml"`` is byte-identical to the
    previous shape. A substring test would tolerate stray characters
    (extra whitespace, reordering); this test pins the exact bytes by
    hand-constructing the pre-fix assembly and asserting equality.
    """
    config = _kb_config()
    config["prompt_envelope"] = "xml"

    bot = await DynaBot.from_config(config)

    message = await bot._build_message_with_context("How do I configure memory?")

    # Pre-fix assembly (from the legacy `_build_message_with_context`):
    #
    #     contexts = [
    #         "<knowledge_base>\n" + kb_body + "\n</knowledge_base>",
    #     ]
    #     # memory absent in this config
    #     context_section = "\n\n".join(contexts)
    #     return f"{context_section}\n\n<question>\n{message}\n</question>"
    #
    # Reconstruct it from the kb body the bot's KB layer would have
    # produced, so the assertion holds against whatever the formatter
    # emits (test-doc content can evolve without breaking this test).
    assert bot.knowledge_base is not None
    kb_results = await bot.knowledge_base.query(
        "How do I configure memory?", k=5,
    )
    assert kb_results, "test_docs must yield at least one hit"
    kb_body = bot.knowledge_base.format_context(kb_results, wrap_in_tags=False)
    expected = (
        f"<knowledge_base>\n{kb_body}\n</knowledge_base>\n\n"
        "<question>\nHow do I configure memory?\n</question>"
    )
    assert message == expected


def test_xml_envelope_synthesis_prompt_byte_identical() -> None:
    """The XML-envelope synthesis prompt is byte-for-byte equal to a hand-built legacy assembly.

    The grounded reasoning module previously emitted a literal
    ``"\\n\\n<knowledge_base>\\n{kb_context}\\n</knowledge_base>"``
    block as one of the parts joined with ``"\\n"``. After the refactor,
    the envelope renders the same bytes. This test pins the full
    assembled prompt — not a substring — so a future drift in
    separators, ordering, or kb_block prefix is caught.
    """
    from dataknobs_bots.reasoning.grounded import GroundedReasoning
    from dataknobs_bots.reasoning.grounded_config import (
        GroundedReasoningConfig,
        GroundedSynthesisConfig,
    )

    cfg = GroundedReasoningConfig(
        synthesis=GroundedSynthesisConfig(
            require_citations=False,
            citation_format="section",
            allow_parametric=False,
        ),
    )
    strategy = GroundedReasoning.from_config(
        cfg,
        prompt_envelope=PromptEnvelope(PromptEnvelopeStyle.XML),
    )
    prompt = strategy.build_synthesis_system_prompt(
        "KB body here", "Original system prompt",
    )

    # Pre-fix inline assembly (from `build_synthesis_system_prompt`):
    #
    #     parts = [original_system_prompt]
    #     parts.append(
    #         "\n\n<knowledge_base>\n" + kb_context + "\n</knowledge_base>"
    #     )
    #     # require_citations=False, allow_parametric=False produces:
    #     parts.append(
    #         "\nBase your response on the knowledge base content provided above. "
    #         "If the knowledge base content does not contain sufficient "
    #         "information to fully answer the question, explicitly state "
    #         "what is missing. Do not fill gaps with information from "
    #         "outside the knowledge base."
    #     )
    #     return "\n".join(parts)
    #
    # The "\n".join inserts one "\n" between each part. The kb_block
    # already starts with "\n\n", so the gap between
    # `original_system_prompt` and `<knowledge_base>` is three
    # newlines — a documented quirk of the pre-fix assembly that we
    # preserve byte-for-byte under XML.
    kb_block = "\n\n<knowledge_base>\nKB body here\n</knowledge_base>"
    grounding_text = (
        "\nBase your response on the knowledge base content provided "
        "above. If the knowledge base content does not contain sufficient "
        "information to fully answer the question, explicitly state "
        "what is missing. Do not fill gaps with information from "
        "outside the knowledge base."
    )
    expected = "\n".join(["Original system prompt", kb_block, grounding_text])

    assert prompt == expected

"""Prompt envelope — one helper for labeled context blocks.

A :class:`PromptEnvelope` renders a labeled context section ("Knowledge
base", "Conversation history", "Question", ...) in one of three styles:
markdown headings, XML tags, or a bare-label prose form.

Every site that today inlines a fixed envelope shape (the bot's
auto-context user-prompt assembler, the knowledge-base layer, the RAG
context formatter, and the grounded-reasoning synthesis system prompt)
goes through this one helper so the style decision lives in one place
and cannot drift across sites.

Why this exists:
    Small instruction-tuned models tend to complete an XML-wrapped
    input shape by emitting a matching wrapper element around their
    reply (e.g. emitting ``<response>...</response>``). The assistant's
    wrapped output then persists in conversation history and seeds the
    next turn — self-reinforcing. Switching the default envelope to
    markdown removes the mirroring cue; consumers that depend on the
    legacy XML shape can pin the ``xml`` style on ``DynaBotConfig``.

Canonical sections:
    Three sections are rendered repeatedly across the codebase — the
    knowledge base, conversation history, and the user's question. Their
    ``(label, tag)`` pairs live in :data:`SECTION_LABELS` so a rename
    is a one-line change. Use the named helpers
    (:meth:`PromptEnvelope.knowledge_base_section` etc.) or
    :meth:`PromptEnvelope.section_for_tag` rather than spelling the
    pair at the call site.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType


class PromptEnvelopeStyle(str, Enum):
    r"""Available envelope styles.

    - :attr:`MARKDOWN` — ``## Label\n\nbody`` sections separated by
      ``\n\n---\n\n``. The default. Visibly bounded for the model
      without inviting tag-mirroring completion.
    - :attr:`XML` — ``<tag>\nbody\n</tag>`` blocks separated by
      ``\n\n``. The legacy shape; kept as an opt-in escape hatch.
    - :attr:`PROSE` — ``Label:\n\nbody`` blocks separated by ``\n\n``.
      An even-more-conservative fallback for models that over-formalize
      on ``##`` headings.
    """

    MARKDOWN = "markdown"
    XML = "xml"
    PROSE = "prose"


# Canonical ``(tag, label)`` pairs for the sections the bot, KB layer,
# and grounded reasoning render. Spelling the pair once here is what
# keeps :meth:`PromptEnvelope.knowledge_base_section` /
# :meth:`PromptEnvelope.conversation_history_section` /
# :meth:`PromptEnvelope.question_section` consistent with the formatter's
# tag-driven path (:meth:`PromptEnvelope.section_for_tag`). A rename is
# a one-line change here, not a sweep through every call site.
SECTION_LABELS: MappingProxyType[str, str] = MappingProxyType(
    {
        "knowledge_base": "Knowledge base",
        "conversation_history": "Conversation history",
        "question": "Question",
    }
)


def _label_for_tag(tag: str) -> str:
    """Resolve a tag to its human-readable label.

    Canonical tags are looked up in :data:`SECTION_LABELS`. Unknown
    tags fall back to a derivation (``"foo_bar"`` → ``"Foo bar"``) so
    direct callers of :meth:`ContextFormatter.wrap_for_prompt` with a
    custom tag still get a sensible label without having to register
    it. New first-class sections should be added to
    :data:`SECTION_LABELS`.
    """
    canonical = SECTION_LABELS.get(tag)
    if canonical is not None:
        return canonical
    return tag.replace("_", " ").capitalize()


class PromptEnvelope:
    """Render a labeled context block in a consistent style.

    A small immutable object: construct one per bot (the style choice
    is bot-wide), pass it to the four sites that build context blocks,
    and call :meth:`section` per block + :meth:`joiner` between them.

    For the three canonical sections, prefer the named helpers
    (:meth:`knowledge_base_section`, :meth:`conversation_history_section`,
    :meth:`question_section`) so the ``(label, tag)`` pair is not
    duplicated at every call site.
    """

    __slots__ = ("_style",)

    def __init__(
        self,
        style: PromptEnvelopeStyle = PromptEnvelopeStyle.MARKDOWN,
    ) -> None:
        self._style = style

    @property
    def style(self) -> PromptEnvelopeStyle:
        """The style this envelope renders in."""
        return self._style

    def section(self, label: str, body: str, *, tag: str) -> str:
        """Render one labeled context section.

        Args:
            label: Human-readable heading used by the markdown and prose
                styles (e.g. ``"Knowledge base"``).
            body: The section's text content. An empty body yields an
                empty string regardless of style.
            tag: XML element name used by the XML style. Every caller
                supplies both ``label`` and ``tag`` so a style change is
                a no-op at the call sites.

        Returns:
            The rendered section, or ``""`` if ``body`` is empty.

        Note:
            For the canonical sections (knowledge base, conversation
            history, question), prefer the named helpers — they pull
            the ``(label, tag)`` pair from :data:`SECTION_LABELS`
            instead of having callers spell it out.
        """
        if not body:
            return ""
        if self._style is PromptEnvelopeStyle.XML:
            return f"<{tag}>\n{body}\n</{tag}>"
        if self._style is PromptEnvelopeStyle.MARKDOWN:
            return f"## {label}\n\n{body}"
        return f"{label}:\n\n{body}"

    def section_for_tag(self, tag: str, body: str) -> str:
        """Render a section keyed by ``tag`` alone.

        The label is resolved through :data:`SECTION_LABELS`
        (falling back to a ``"foo_bar"`` → ``"Foo bar"`` derivation
        for unknown tags), so the formatter and other generic callers
        do not need to keep their own tag→label table in sync with the
        canonical one.
        """
        return self.section(_label_for_tag(tag), body, tag=tag)

    def knowledge_base_section(self, body: str) -> str:
        """Render the knowledge-base section in the envelope's style.

        Canonical pair: ``("Knowledge base", "knowledge_base")``.
        """
        return self.section_for_tag("knowledge_base", body)

    def conversation_history_section(self, body: str) -> str:
        """Render the conversation-history section in the envelope's style.

        Canonical pair: ``("Conversation history", "conversation_history")``.
        """
        return self.section_for_tag("conversation_history", body)

    def question_section(self, body: str) -> str:
        """Render the user-question section in the envelope's style.

        Canonical pair: ``("Question", "question")``.
        """
        return self.section_for_tag("question", body)

    def joiner(self) -> str:
        """Return the separator placed between rendered sections."""
        if self._style is PromptEnvelopeStyle.MARKDOWN:
            return "\n\n---\n\n"
        return "\n\n"


__all__ = ["PromptEnvelope", "PromptEnvelopeStyle", "SECTION_LABELS"]

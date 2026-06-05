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
"""

from __future__ import annotations

from enum import Enum


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


class PromptEnvelope:
    """Render a labeled context block in a consistent style.

    A small immutable object: construct one per bot (the style choice
    is bot-wide), pass it to the four sites that build context blocks,
    and call :meth:`section` per block + :meth:`joiner` between them.
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
        """
        if not body:
            return ""
        if self._style is PromptEnvelopeStyle.XML:
            return f"<{tag}>\n{body}\n</{tag}>"
        if self._style is PromptEnvelopeStyle.MARKDOWN:
            return f"## {label}\n\n{body}"
        return f"{label}:\n\n{body}"

    def joiner(self) -> str:
        """Return the separator placed between rendered sections."""
        if self._style is PromptEnvelopeStyle.MARKDOWN:
            return "\n\n---\n\n"
        return "\n\n"


__all__ = ["PromptEnvelope", "PromptEnvelopeStyle"]

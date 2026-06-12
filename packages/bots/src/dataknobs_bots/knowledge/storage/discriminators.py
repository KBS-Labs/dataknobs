"""Discriminator adapters for knowledge resource backends.

Wraps a backend's :meth:`KnowledgeResourceBackend.classify_key` method
as a generic :class:`~dataknobs_common.discriminator.Discriminator`
``[str, KnowledgeKeyKind]`` so consumer code composing event-routing
logic can dispatch through the protocol shape rather than coupling
directly to the backend interface.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataknobs_bots.knowledge.storage.backend import KnowledgeResourceBackend
from dataknobs_bots.knowledge.storage.key_layout import KnowledgeKeyKind


@dataclass(frozen=True)
class BackendKeyDiscriminator:
    """Adapt a :class:`KnowledgeResourceBackend` to the generic
    :class:`~dataknobs_common.discriminator.Discriminator`
    ``[str, KnowledgeKeyKind]`` protocol.

    Use when composing backend-key classification with other
    discriminators (payload-field, intent, etc.) through the generic
    protocol shape without coupling consumer code to the backend
    interface directly.

    Example:
        >>> backend = FileKnowledgeBackend(base_path="/srv/kb")
        >>> discriminator = BackendKeyDiscriminator(backend)
        >>> discriminator.classify("kb1/content/doc1.pdf")
        <KnowledgeKeyKind.CONTENT: 'content'>

    The wrapped backend's ``classify_key`` method is called on every
    ``classify`` invocation; no caching is applied. Consumers needing
    caching wrap this discriminator in a caching decorator at the
    consumer's choice.

    The ``frozen=True`` dataclass decoration provides ``__eq__`` and
    ``__hash__`` keyed on the wrapped backend, so two adapters wrapping
    the same backend instance compare equal — useful for adapter-cache
    lookups.
    """

    backend: KnowledgeResourceBackend

    def classify(self, value: str) -> KnowledgeKeyKind:
        return self.backend.classify_key(value)


__all__ = ["BackendKeyDiscriminator"]

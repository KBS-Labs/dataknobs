"""End-to-end composition test: ``MultiFieldDiscriminator`` routes on
``backend.classify_key`` + a payload-field classifier.

Co-located with :mod:`test_backend_key_discriminator` (which pins the
adapter conformance) so the canonical event-router shape — combining
the backend classifier with a payload-field classifier — has its
regression guard alongside the per-component guards.
"""

from __future__ import annotations

from pathlib import Path

from dataknobs_common.discriminator import (
    MappingDiscriminator,
    MultiFieldDiscriminator,
)

from dataknobs_bots.knowledge.storage import (
    BackendKeyDiscriminator,
    FileKnowledgeBackend,
    KnowledgeKeyKind,
)


def test_multi_field_composes_with_backend_key_discriminator(
    tmp_path: Path,
) -> None:
    """A consumer wiring multi-aspect event routing classifies the
    backend key kind AND a payload label through one composable surface."""
    backend = FileKnowledgeBackend(base_path=tmp_path)
    multi = MultiFieldDiscriminator({
        "key": BackendKeyDiscriminator(backend),
        "label": MappingDiscriminator(
            mapping={"high": "critical", "low": "deferred"},
            default="normal",
        ),
    })

    result = multi.classify({
        "key": "kb1/content/doc.pdf",
        "label": "high",
    })
    assert result == {
        "key": KnowledgeKeyKind.CONTENT,
        "label": "critical",
    }


def test_multi_field_missing_payload_field_returns_none(
    tmp_path: Path,
) -> None:
    """Missing payload fields surface as None in the result dict — the
    consumer's dispatch logic can distinguish 'absent' from 'classified
    as None'."""
    backend = FileKnowledgeBackend(base_path=tmp_path)
    multi = MultiFieldDiscriminator({
        "key": BackendKeyDiscriminator(backend),
        "label": MappingDiscriminator(
            mapping={"high": "critical"},
            default="normal",
        ),
    })

    result = multi.classify({"key": "kb1/_metadata.json"})
    assert result == {
        "key": KnowledgeKeyKind.METADATA,
        "label": None,
    }

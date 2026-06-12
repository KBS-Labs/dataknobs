"""Tests for :class:`BackendKeyDiscriminator`.

Co-located with :mod:`test_key_layout` (which pins the underlying
``classify_key`` contract) so an adapter behavior regression and a
contract regression surface in the same test module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_bots.knowledge.storage import (
    BackendKeyDiscriminator,
    FileKnowledgeBackend,
    InMemoryKnowledgeBackend,
    KnowledgeKeyKind,
)
from dataknobs_bots.knowledge.storage.s3 import S3KnowledgeBackend
from dataknobs_common.discriminator import Discriminator


@pytest.fixture
def file_backend(tmp_path: Path) -> FileKnowledgeBackend:
    return FileKnowledgeBackend(base_path=tmp_path)


@pytest.fixture
def memory_backend() -> InMemoryKnowledgeBackend:
    return InMemoryKnowledgeBackend()


@pytest.fixture
def s3_backend() -> S3KnowledgeBackend:
    # classify_key is pure string manipulation; the S3 client is not
    # initialized here, so no live AWS / moto session is required.
    return S3KnowledgeBackend(bucket="test-bucket", prefix="kb/")


def test_adapter_conforms_to_discriminator_protocol(
    file_backend: FileKnowledgeBackend,
) -> None:
    discriminator = BackendKeyDiscriminator(file_backend)
    assert isinstance(discriminator, Discriminator)


def test_adapter_classifies_content_key(
    file_backend: FileKnowledgeBackend,
) -> None:
    discriminator = BackendKeyDiscriminator(file_backend)
    assert (
        discriminator.classify("kb1/content/doc1.pdf")
        is KnowledgeKeyKind.CONTENT
    )


def test_adapter_classifies_metadata_key(
    file_backend: FileKnowledgeBackend,
) -> None:
    discriminator = BackendKeyDiscriminator(file_backend)
    assert (
        discriminator.classify("kb1/_metadata.json")
        is KnowledgeKeyKind.METADATA
    )


def test_adapter_classifies_snapshot_key(
    file_backend: FileKnowledgeBackend,
) -> None:
    discriminator = BackendKeyDiscriminator(file_backend)
    assert (
        discriminator.classify("kb1/_snapshots/v1.json")
        is KnowledgeKeyKind.SNAPSHOT
    )


def test_adapter_classifies_unknown_key(
    file_backend: FileKnowledgeBackend,
) -> None:
    discriminator = BackendKeyDiscriminator(file_backend)
    assert discriminator.classify("stray/file") is KnowledgeKeyKind.UNKNOWN


def test_adapter_classifies_consistent_with_backend(
    file_backend: FileKnowledgeBackend,
) -> None:
    """The adapter MUST return what ``backend.classify_key`` returns —
    the adapter is a pure projection, never a re-implementation."""
    discriminator = BackendKeyDiscriminator(file_backend)
    test_keys = [
        "kb1/content/foo.pdf",
        "kb1/content/sub/_metadata.json",  # CONTENT wins (segment precedence)
        "kb1/_metadata.json",
        "kb1/_snapshots/v2.json",
        "stray/file",
        "",
    ]
    for key in test_keys:
        assert discriminator.classify(key) is file_backend.classify_key(key)


@pytest.mark.parametrize("backend_name", ["file", "memory", "s3"])
def test_adapter_works_for_all_in_tree_backends(
    backend_name: str,
    file_backend: FileKnowledgeBackend,
    memory_backend: InMemoryKnowledgeBackend,
    s3_backend: S3KnowledgeBackend,
) -> None:
    """Every in-tree backend honors the same key layout via the mixin's
    canonical ``classify_key`` — the adapter therefore works across all
    three without backend-specific branches."""
    backend = {
        "file": file_backend,
        "memory": memory_backend,
        "s3": s3_backend,
    }[backend_name]
    discriminator = BackendKeyDiscriminator(backend)
    assert (
        discriminator.classify("kb1/content/x")
        is KnowledgeKeyKind.CONTENT
    )
    assert (
        discriminator.classify("kb1/_metadata.json")
        is KnowledgeKeyKind.METADATA
    )
    assert (
        discriminator.classify("kb1/_snapshots/v1.json")
        is KnowledgeKeyKind.SNAPSHOT
    )


def test_adapter_equality_on_same_backend(
    file_backend: FileKnowledgeBackend,
) -> None:
    """``frozen=True`` dataclass: two adapters wrapping the same
    backend instance compare equal (useful for adapter-cache lookups)."""
    d1 = BackendKeyDiscriminator(file_backend)
    d2 = BackendKeyDiscriminator(file_backend)
    assert d1 == d2
    assert hash(d1) == hash(d2)


def test_adapter_inequality_on_different_backends(
    file_backend: FileKnowledgeBackend,
    memory_backend: InMemoryKnowledgeBackend,
) -> None:
    d1 = BackendKeyDiscriminator(file_backend)
    d2 = BackendKeyDiscriminator(memory_backend)
    assert d1 != d2

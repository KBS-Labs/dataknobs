"""Cross-backend conformance tests for the key-layout contract.

Pins :meth:`KnowledgeResourceBackend.classify_key` and
:meth:`KnowledgeResourceBackend.key_pattern` against every in-tree
backend so a future drift in the layout constants or key-derivation
methods is caught by the conformance suite before it silently breaks a
consumer's external-event-source filter.

``key_pattern`` and ``classify_key`` are pure string operations — they
inspect the configured prefix and the key's path segments and never
touch S3 — so the S3 leg of those conformance checks builds an
*uninitialized* backend with no AWS service at all. The single
end-to-end leg that actually writes objects
(:func:`test_feedback_loop_reproduction_s3`) runs against real
LocalStack (``bin/dk up``) and skips when it is unavailable; moto's
``mock_aws`` is incompatible with the aioboto3 transport.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from dataknobs_bots.knowledge.storage import (
    FileKnowledgeBackend,
    InMemoryKnowledgeBackend,
    KnowledgeKeyKind,
    KnowledgeResourceBackendMixin,
)
from dataknobs_bots.knowledge.storage.s3 import S3KnowledgeBackend
from dataknobs_common.testing import requires_localstack

BUCKET = "kb-key-layout-bucket"
PREFIX = "rag/"


# ---------------------------------------------------------------------------
# Helpers: build each in-tree backend for the classify_key parametrize
# ---------------------------------------------------------------------------


def _make_memory_backend() -> InMemoryKnowledgeBackend:
    return InMemoryKnowledgeBackend()


def _make_file_backend(tmp: Path) -> FileKnowledgeBackend:
    return FileKnowledgeBackend(base_path=tmp / "kb")


def _make_s3_backend() -> S3KnowledgeBackend:
    """An uninitialized S3 backend — key_pattern/classify_key are pure.

    No bucket, no client, no AWS service: the layout methods under test
    derive their answers from the configured prefix and the key's path
    segments alone.
    """
    return S3KnowledgeBackend(bucket=BUCKET, prefix=PREFIX)


# ---------------------------------------------------------------------------
# T1 — classify_key cross-backend conformance
# ---------------------------------------------------------------------------


# Inputs are layout-shape-agnostic — classify_key inspects path segments,
# not the backend's specific prefix. One canonical key table covers every
# backend (the segment-precedence rule is the same).
CLASSIFY_CASES: list[tuple[str, KnowledgeKeyKind]] = [
    ("foo/content/bar.md", KnowledgeKeyKind.CONTENT),
    ("foo/content/sub/deep.md", KnowledgeKeyKind.CONTENT),
    # A consumer file legitimately named _metadata.json (a real markdown
    # file with that name is valid content). The leading content/ ancestor
    # wins — the helper must NOT false-positive as METADATA.
    ("foo/content/_metadata.json", KnowledgeKeyKind.CONTENT),
    ("foo/_metadata.json", KnowledgeKeyKind.METADATA),
    ("foo/_snapshots/abc123.json", KnowledgeKeyKind.SNAPSHOT),
    ("other/junk.txt", KnowledgeKeyKind.UNKNOWN),
]


@pytest.mark.parametrize("raw_key, expected", CLASSIFY_CASES)
def test_classify_key_memory(
    raw_key: str, expected: KnowledgeKeyKind
) -> None:
    backend = _make_memory_backend()
    assert backend.classify_key(raw_key) == expected


@pytest.mark.parametrize("raw_key, expected", CLASSIFY_CASES)
def test_classify_key_file(
    raw_key: str, expected: KnowledgeKeyKind, tmp_path: Path
) -> None:
    backend = _make_file_backend(tmp_path)
    assert backend.classify_key(raw_key) == expected


@pytest.mark.parametrize("raw_key, expected", CLASSIFY_CASES)
def test_classify_key_s3(
    raw_key: str, expected: KnowledgeKeyKind
) -> None:
    backend = _make_s3_backend()
    assert backend.classify_key(raw_key) == expected


# ---------------------------------------------------------------------------
# T2 / T3 — key_pattern(CONTENT) all-domain + single-domain conformance
# ---------------------------------------------------------------------------


def test_key_pattern_s3_content_all_domains() -> None:
    backend = _make_s3_backend()
    assert backend.key_pattern() == f"{PREFIX}*/content/*"
    # default kind is CONTENT, default domain_id is None — same answer
    assert (
        backend.key_pattern(KnowledgeKeyKind.CONTENT)
        == f"{PREFIX}*/content/*"
    )


def test_key_pattern_s3_content_single_domain() -> None:
    backend = _make_s3_backend()
    assert (
        backend.key_pattern(KnowledgeKeyKind.CONTENT, domain_id="acme")
        == f"{PREFIX}acme/content/*"
    )


def test_key_pattern_file_content_all_domains(tmp_path: Path) -> None:
    backend = _make_file_backend(tmp_path)
    base = str(tmp_path / "kb")
    assert backend.key_pattern() == f"{base}/*/content/**"


def test_key_pattern_file_content_single_domain(tmp_path: Path) -> None:
    backend = _make_file_backend(tmp_path)
    base = str(tmp_path / "kb")
    assert (
        backend.key_pattern(KnowledgeKeyKind.CONTENT, domain_id="acme")
        == f"{base}/acme/content/**"
    )


def test_key_pattern_memory_returns_empty_sentinel() -> None:
    backend = _make_memory_backend()
    # No event-source filter is meaningful in-process; empty-string
    # sentinel is the protocol-symmetry contract.
    assert backend.key_pattern() == ""
    assert backend.key_pattern(KnowledgeKeyKind.METADATA) == ""
    assert backend.key_pattern(
        KnowledgeKeyKind.SNAPSHOT, domain_id="acme"
    ) == ""


# ---------------------------------------------------------------------------
# T4 — key_pattern(METADATA) / key_pattern(SNAPSHOT) match the actual keys
# ---------------------------------------------------------------------------
#
# This is the layout-drift pin: a future change to the private constants
# or key-derivation methods must keep the pattern in sync. We check
# pattern STRINGS against the private key-derivation helpers (a stronger
# assertion than just "looks reasonable").


def test_key_pattern_s3_metadata_matches_metadata_key() -> None:
    backend = _make_s3_backend()
    # The all-domains pattern must equal the single-domain pattern
    # with the wildcard substituted for the domain segment, and that
    # equals what _metadata_key produces for that domain.
    pattern_acme = backend.key_pattern(
        KnowledgeKeyKind.METADATA, domain_id="acme"
    )
    assert pattern_acme == backend._metadata_key("acme")
    # Wildcard pattern is the same shape with * for the domain.
    assert backend.key_pattern(KnowledgeKeyKind.METADATA) == (
        f"{PREFIX}*/_metadata.json"
    )


def test_key_pattern_s3_snapshot_matches_snapshot_key() -> None:
    backend = _make_s3_backend()
    # The snapshot pattern matches the prefix every snapshot key
    # lives under.
    snap_key = backend._snapshot_key("acme", "deadbeef")
    prefix = (
        f"{PREFIX}acme/_snapshots/"
    )  # what the * in the pattern stands in for
    assert snap_key.startswith(prefix)
    assert backend.key_pattern(
        KnowledgeKeyKind.SNAPSHOT, domain_id="acme"
    ) == f"{prefix}*"
    assert backend.key_pattern(KnowledgeKeyKind.SNAPSHOT) == (
        f"{PREFIX}*/_snapshots/*"
    )


def test_key_pattern_file_metadata_matches_metadata_path(
    tmp_path: Path,
) -> None:
    backend = _make_file_backend(tmp_path)
    base = str(tmp_path / "kb")
    pattern_acme = backend.key_pattern(
        KnowledgeKeyKind.METADATA, domain_id="acme"
    )
    assert pattern_acme == f"{base}/acme/_metadata.json"
    assert pattern_acme == str(backend._metadata_path("acme"))


def test_key_pattern_file_snapshot_matches_snapshot_path(
    tmp_path: Path,
) -> None:
    backend = _make_file_backend(tmp_path)
    base = str(tmp_path / "kb")
    snap_path = backend._snapshot_file("acme", "deadbeef")
    prefix = f"{base}/acme/_snapshots/"
    assert str(snap_path).startswith(prefix)
    assert backend.key_pattern(
        KnowledgeKeyKind.SNAPSHOT, domain_id="acme"
    ) == f"{prefix}*"


# ---------------------------------------------------------------------------
# T5 — End-to-end feedback-loop reproduction (the headline test)
# ---------------------------------------------------------------------------
#
# Enumerate every key the backend writes during one put_file (the smallest
# realistic ingest cycle) and assert: the consumer's CONTENT key is the
# ONLY one matched by the CONTENT pattern, and every DK-managed state
# write classifies as METADATA or SNAPSHOT. This is the closest a
# unit-level test can come to reproducing the positive-feedback-loop
# pattern without running a real EventBridge rule.


def _glob_match(pattern: str, key: str) -> bool:
    """Match a glob-shaped pattern against a key.

    ``**`` matches across path separators; ``*`` matches within a
    segment. Used for asserting CONTENT-pattern matching on real keys.
    """
    import fnmatch

    if "**" not in pattern:
        return fnmatch.fnmatchcase(key, pattern)
    # Translate ``**`` to a fnmatch-friendly any-character wildcard.
    # fnmatch ``*`` already crosses path separators in our usage, so the
    # simplest faithful translation is to drop the second ``*``.
    return fnmatch.fnmatchcase(key, pattern.replace("**", "*"))


@pytest.mark.integration
@pytest.mark.s3
@requires_localstack
async def test_feedback_loop_reproduction_s3(s3_kb_config) -> None:
    """A single put_file produces one CONTENT key and at least one
    state key on S3 — confirm the CONTENT pattern matches only the
    consumer key.
    """
    backend = S3KnowledgeBackend.from_config(s3_kb_config)
    await backend.initialize()
    try:
        await backend.create_kb("acme")
        await backend.put_file("acme", "intro.md", b"# Hello")

        # Enumerate every object under the domain prefix using the
        # backend's own (aioboto3) session.
        domain_prefix = f"{backend._prefix}acme/"
        async with backend._session.client(
            "s3", **backend._client_kwargs
        ) as s3:
            response = await s3.list_objects_v2(
                Bucket=backend._bucket, Prefix=domain_prefix
            )
        keys = [obj["Key"] for obj in response.get("Contents", [])]
        assert keys, "expected at least one key after put_file"

        pattern = backend.key_pattern(
            KnowledgeKeyKind.CONTENT, domain_id="acme"
        )

        content_keys = [
            k for k in keys if backend.classify_key(k) is KnowledgeKeyKind.CONTENT
        ]
        metadata_keys = [
            k for k in keys if backend.classify_key(k) is KnowledgeKeyKind.METADATA
        ]
        snapshot_keys = [
            k for k in keys if backend.classify_key(k) is KnowledgeKeyKind.SNAPSHOT
        ]

        assert len(content_keys) == 1, (
            f"expected exactly one CONTENT key; got {content_keys}"
        )
        assert _glob_match(pattern, content_keys[0]), (
            f"CONTENT pattern {pattern!r} must match the content key "
            f"{content_keys[0]!r}"
        )
        assert metadata_keys, (
            "expected at least one METADATA write during put_file "
            "(the feedback-loop trigger that broke production)"
        )
        for state_key in metadata_keys + snapshot_keys:
            assert not _glob_match(pattern, state_key), (
                f"CONTENT pattern {pattern!r} must NOT match state "
                f"key {state_key!r} — that's the feedback-loop bug"
            )
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_feedback_loop_reproduction_file(tmp_path: Path) -> None:
    """Same as the S3 case but for the filesystem backend.

    Walks the on-disk tree, classifies every file, asserts the CONTENT
    glob matches only the consumer file.
    """
    backend = _make_file_backend(tmp_path)
    await backend.initialize()
    try:
        await backend.create_kb("acme")
        await backend.put_file("acme", "intro.md", b"# Hello")

        # Enumerate every file under the domain directory.
        base = tmp_path / "kb"
        domain_root = base / "acme"
        keys = [str(p) for p in domain_root.rglob("*") if p.is_file()]
        assert keys, "expected at least one file after put_file"

        pattern = backend.key_pattern(
            KnowledgeKeyKind.CONTENT, domain_id="acme"
        )

        content_keys = [
            k for k in keys if backend.classify_key(k) is KnowledgeKeyKind.CONTENT
        ]
        metadata_keys = [
            k for k in keys if backend.classify_key(k) is KnowledgeKeyKind.METADATA
        ]
        snapshot_keys = [
            k for k in keys if backend.classify_key(k) is KnowledgeKeyKind.SNAPSHOT
        ]

        assert len(content_keys) == 1, (
            f"expected exactly one CONTENT key; got {content_keys}"
        )
        assert _glob_match(pattern, content_keys[0]), (
            f"CONTENT pattern {pattern!r} must match the content key "
            f"{content_keys[0]!r}"
        )
        assert metadata_keys, (
            "expected at least one METADATA write during put_file"
        )
        for state_key in metadata_keys + snapshot_keys:
            assert not _glob_match(pattern, state_key), (
                f"CONTENT pattern {pattern!r} must NOT match state "
                f"key {state_key!r}"
            )
    finally:
        await backend.close()


# ---------------------------------------------------------------------------
# T6 — classify_key segment-wins-over-suffix edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, expected",
    [
        # content/ segment wins, even though _snapshots appears later.
        ("foo/content/bar/_snapshots/x.json", KnowledgeKeyKind.CONTENT),
        # content/ segment wins regardless of position.
        ("foo/_snapshots/content/x.json", KnowledgeKeyKind.CONTENT),
        # Terminal segment + no content/ ancestor → METADATA.
        ("foo/_metadata.json", KnowledgeKeyKind.METADATA),
        # Pathological case: _metadata.json is not the terminal segment.
        ("foo/_metadata.json/extra", KnowledgeKeyKind.UNKNOWN),
    ],
)
def test_classify_key_segment_precedence_edges(
    key: str, expected: KnowledgeKeyKind
) -> None:
    """Pins the 'any content/ segment wins' rule explicitly."""
    backend = _make_memory_backend()  # rule is layout-agnostic
    assert backend.classify_key(key) == expected


def test_classify_key_empty_or_root_key() -> None:
    """An empty or root-only key classifies as UNKNOWN (defensive)."""
    backend = _make_memory_backend()
    assert backend.classify_key("") == KnowledgeKeyKind.UNKNOWN
    assert backend.classify_key("/") == KnowledgeKeyKind.UNKNOWN


# ---------------------------------------------------------------------------
# T7 — Constants live at the mixin (Change B regression pin)
# ---------------------------------------------------------------------------


def test_constants_are_declared_at_the_mixin() -> None:
    """The canonical layout constants live once, on the mixin."""
    assert KnowledgeResourceBackendMixin.METADATA_FILE == "_metadata.json"
    assert KnowledgeResourceBackendMixin.CONTENT_DIR == "content"
    assert KnowledgeResourceBackendMixin.SNAPSHOTS_DIR == "_snapshots"


def test_each_backend_inherits_mixin_constants(tmp_path: Path) -> None:
    """Every in-tree backend resolves the constants via MRO to the mixin."""
    backends = [
        _make_memory_backend(),
        _make_file_backend(tmp_path),
        _make_s3_backend(),
    ]
    for backend in backends:
        assert (
            backend.METADATA_FILE
            == KnowledgeResourceBackendMixin.METADATA_FILE
        )
        assert (
            backend.CONTENT_DIR
            == KnowledgeResourceBackendMixin.CONTENT_DIR
        )
        assert (
            backend.SNAPSHOTS_DIR
            == KnowledgeResourceBackendMixin.SNAPSHOTS_DIR
        )


# ---------------------------------------------------------------------------
# T8 — Out-of-tree backend gets classify_key for free via the mixin
# ---------------------------------------------------------------------------


class _OutOfTreeBackend(KnowledgeResourceBackendMixin):
    """Minimal sibling backend implementing none of the storage methods.

    Used purely to assert consumer-extensibility: a third-party backend
    that honors the documented layout gets correct classification for
    free by mixing in :class:`KnowledgeResourceBackendMixin`.
    """


@pytest.mark.parametrize("raw_key, expected", CLASSIFY_CASES)
def test_out_of_tree_backend_inherits_classify_key(
    raw_key: str, expected: KnowledgeKeyKind
) -> None:
    backend = _OutOfTreeBackend()
    assert backend.classify_key(raw_key) == expected


# ---------------------------------------------------------------------------
# T9 — key_pattern raises on UNKNOWN kind (fail closed)
# ---------------------------------------------------------------------------


def test_key_pattern_s3_raises_on_unknown_kind() -> None:
    backend = _make_s3_backend()
    with pytest.raises(ValueError, match="UNKNOWN"):
        backend.key_pattern(kind=KnowledgeKeyKind.UNKNOWN)


def test_key_pattern_file_raises_on_unknown_kind(tmp_path: Path) -> None:
    backend = _make_file_backend(tmp_path)
    with pytest.raises(ValueError, match="UNKNOWN"):
        backend.key_pattern(kind=KnowledgeKeyKind.UNKNOWN)


def test_key_pattern_memory_returns_empty_for_unknown() -> None:
    """In-memory backend tolerates UNKNOWN (no pattern-matching surface).

    Consistent with the in-memory key_pattern returning empty for every
    kind — the method exists for protocol symmetry only.
    """
    backend = _make_memory_backend()
    assert backend.key_pattern(kind=KnowledgeKeyKind.UNKNOWN) == ""


# ---------------------------------------------------------------------------
# Hygiene — no leftover tempdirs (covered by tmp_path fixture; sanity pin)
# ---------------------------------------------------------------------------


def test_tempdir_isolation_does_not_leak(tmp_path: Path) -> None:
    """Sanity check: a backend built from tmp_path uses the per-test root."""
    backend = _make_file_backend(tmp_path)
    assert str(tmp_path) in backend.key_pattern(domain_id="acme")
    # Ensure tempfile module is still importable (sanity: nothing
    # exotic broke imports).
    with tempfile.TemporaryDirectory() as td:
        assert Path(td).exists()

"""A knowledge base's namespace and the tenant-state namespace are disjoint.

The segment rule makes ``domain_id`` occupy exactly one slot. It does not
make that slot *disjoint from the constants the layout puts beside it* —
and the tenant state prefix contributes a constant at the same level as a
domain root::

    _s3_key("tenants")                     -> {p}tenants/
    _metadata_key("proj", ctx=bound_acme)  -> {p}tenants/acme/_state/proj/_metadata.json

The second key sits under the first prefix. ``delete_kb`` deletes by
prefix on both persistent backends, so ``delete_kb("tenants")`` — a legal
one-segment name that ``safe_segment`` passes, and whose collision the
prefix's own containment check has no opinion about — destroyed every
tenant's ingest state for every domain in the deployment.

Both guards are satisfied and the slot still collides. That is the same
reading error the segment rule was introduced to close, one level up:
containment asks whether the name left the tree, the segment rule asks
whether it stayed in its own segment, and neither asks whether the
segment it occupies is one the layout had already spoken for.

The fix is structural rather than a reserved-word list. A list cannot
cover :class:`PrefixedTenantContext`, whose prefix — including its first
segment — is a consumer-supplied format string that this package never
sees. So the backend layout roots *all* context-scoped state under one
segment of its own, and refuses a ``domain_id`` that could name it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_bots.knowledge.storage.file import FileKnowledgeBackend
from dataknobs_bots.knowledge.storage.memory import InMemoryKnowledgeBackend
from dataknobs_bots.knowledge.storage.s3 import S3KnowledgeBackend
from dataknobs_common.paths import SegmentEscapeError
from dataknobs_common.tenancy import (
    BoundTenantContext,
    PrefixedTenantContext,
    SharedCorpusTenantContext,
)

BUCKET = "kb-state-namespace-bucket"
PREFIX = "kb/"

#: The first segment of every reference context's state prefix. A domain
#: of this name is what reaches the tenant subtree; it is a legal segment,
#: which is exactly why the segment rule does not stop it.
COLLIDING = "tenants"


@pytest.fixture
def bound() -> BoundTenantContext:
    return BoundTenantContext(tenant_id="acme", domain_id="proj")


@pytest.fixture
async def file_backend(tmp_path: Path) -> FileKnowledgeBackend:
    backend = FileKnowledgeBackend(base_path=tmp_path / "kb")
    await backend.initialize()
    return backend


@pytest.fixture
async def memory_backend() -> InMemoryKnowledgeBackend:
    backend = InMemoryKnowledgeBackend()
    await backend.initialize()
    return backend


def _s3() -> S3KnowledgeBackend:
    """Uninitialized — the key helpers are pure string composition."""
    return S3KnowledgeBackend(bucket=BUCKET, prefix=PREFIX)


class TestNoDomainPrefixContainsAStateKey:
    """The key-level statement of the invariant, on the pure helpers.

    Asserted here rather than only through ``delete_kb`` because the
    prefix relation is the defect; the destructive read of it is one
    consequence, and a later helper composing the same two namespaces
    would reintroduce it without touching ``delete_kb`` at all.
    """

    def test_s3_a_domain_prefix_does_not_contain_tenant_state(
        self, bound: BoundTenantContext
    ) -> None:
        backend = _s3()

        state_key = backend._metadata_key("proj", ctx=bound)

        assert not state_key.startswith(backend._s3_key(COLLIDING))

    def test_s3_a_domain_prefix_does_not_contain_a_tenant_snapshot(
        self, bound: BoundTenantContext
    ) -> None:
        backend = _s3()

        snapshot_key = backend._snapshot_key("proj", "abc123", ctx=bound)

        assert not snapshot_key.startswith(backend._s3_key(COLLIDING))

    def test_file_a_domain_tree_does_not_contain_tenant_state(
        self, tmp_path: Path, bound: BoundTenantContext
    ) -> None:
        backend = FileKnowledgeBackend(base_path=tmp_path / "kb")

        state_path = backend._metadata_path("proj", ctx=bound)

        assert not state_path.is_relative_to(backend._kb_path(COLLIDING))

    def test_the_shared_corpus_context_is_rooted_the_same_way(self) -> None:
        """It emits the identical prefix, so it inherits the identical hole."""
        backend = _s3()
        ctx = SharedCorpusTenantContext(
            tenant_id="acme", domain_id="proj", shared_corpus_id="corpus"
        )

        state_key = backend._metadata_key("proj", ctx=ctx)

        assert not state_key.startswith(backend._s3_key(COLLIDING))

    def test_a_consumer_supplied_prefix_is_rooted_out_of_reach_too(self) -> None:
        """The case a reserved-word list cannot cover.

        ``PrefixedTenantContext`` takes the whole prefix from consumer
        configuration, so its first segment is a name this package never
        sees and could not have enumerated. Rooting the state namespace
        in a segment the layout owns is what makes the guarantee hold
        for a prefix nobody here wrote.
        """
        backend = _s3()
        ctx = PrefixedTenantContext(
            tenant_id="acme", domain_id="proj", prefix_pattern="clients/{tenant_id}/"
        )

        state_key = backend._metadata_key("proj", ctx=ctx)

        assert not state_key.startswith(backend._s3_key("clients"))


class TestTheRootedPrefixAlwaysEndsInASeparator:
    """A prefix without one merged two tenants on the key-string backend.

    The file backend joins path segments and S3 concatenates strings, so
    a ``prefix_pattern`` with no trailing separator behaved differently
    on the two: the file backend produced ``{base}/t-a/bc``, and S3 —
    concatenating prefix, domain and slot — produced ``kb/t-abc/…`` for
    both ``(tenant="a", domain="bc")`` and ``(tenant="ab", domain="c")``.
    Nothing traverses and containment is satisfied; the namespaces merge,
    which is the same failure the segment rule addresses for a name.

    Rooting the prefix now normalises it through a path join and restores
    the separator, so the two stores agree and the pair stays distinct.
    ``PrefixedTenantContext`` still documents pattern ambiguity as the
    consumer's to avoid — this closes only the missing-separator case,
    not a delimiter that can occur inside a tenant id.
    """

    def test_two_tenant_domain_pairs_do_not_merge_on_s3(self) -> None:
        backend = _s3()
        one = PrefixedTenantContext(tenant_id="a", domain_id="bc", prefix_pattern="t-{tenant_id}")
        other = PrefixedTenantContext(tenant_id="ab", domain_id="c", prefix_pattern="t-{tenant_id}")

        assert backend._metadata_key("bc", one) != backend._metadata_key("c", other)

    def test_the_two_backends_separate_them_the_same_way(self, tmp_path: Path) -> None:
        s3 = _s3()
        file_b = FileKnowledgeBackend(base_path=tmp_path / "kb")
        ctx = PrefixedTenantContext(tenant_id="a", domain_id="bc", prefix_pattern="t-{tenant_id}")

        assert s3._metadata_key("bc", ctx).endswith("_scoped/t-a/bc/_metadata.json")
        assert file_b._metadata_path("bc", ctx=ctx).parts[-4:] == (
            "_scoped",
            "t-a",
            "bc",
            "_metadata.json",
        )


class TestTheLayoutReservesItsOwnNamespace:
    """A ``domain_id`` may not name the segment the state namespace roots in."""

    def test_the_scoped_state_root_is_not_an_addressable_domain(self) -> None:
        backend = _s3()

        with pytest.raises(SegmentEscapeError, match="reserved"):
            backend._s3_key(backend.SCOPED_STATE_ROOT)

    @pytest.mark.parametrize("domain_id", ["_scoped", "_state", "_metadata.json", "_snapshots"])
    def test_an_underscore_prefixed_domain_is_refused(self, domain_id: str) -> None:
        """The rule is the ``_`` prefix, not a list of today's slots.

        The layout already spells its own slots inside a knowledge base
        that way (``_metadata.json``, ``_snapshots/``). Reserving the
        prefix rather than the four current names is what keeps a slot
        added later from reopening this by itself.
        """
        backend = _s3()

        with pytest.raises(SegmentEscapeError, match="reserved"):
            backend._s3_key(domain_id)

    @pytest.mark.parametrize("domain_id", ["acme", "acme_content", "a_b", "ACME2", "acme.v2"])
    def test_an_underscore_elsewhere_in_the_name_is_untouched(self, domain_id: str) -> None:
        """Only the leading position is reserved."""
        backend = _s3()

        assert backend._s3_key(domain_id) == f"{PREFIX}{domain_id}/"

    def test_every_backend_refuses_it(self, tmp_path: Path) -> None:
        """A name refused in production is refused where consumers develop."""
        file_b = FileKnowledgeBackend(base_path=tmp_path / "kb")
        memory_b = InMemoryKnowledgeBackend()

        with pytest.raises(SegmentEscapeError, match="reserved"):
            file_b._kb_path("_scoped")
        with pytest.raises(SegmentEscapeError, match="reserved"):
            memory_b._validate_domain_id("_scoped")


class TestDeletingTheCollidingDomainLeavesTenantStateAlone:
    """The reproduced damage: one legal ``delete_kb`` erased every tenant."""

    async def test_file_backend_keeps_tenant_state(
        self, file_backend: FileKnowledgeBackend, bound: BoundTenantContext
    ) -> None:
        """The colliding KB is created first — the realistic ordering.

        A deployment names a knowledge base, then onboards tenants
        against others. Deleting that one knowledge base then took every
        tenant's state for every domain with it, and reported success.
        """
        await file_backend.create_kb("proj")
        await file_backend.create_kb(COLLIDING)
        await file_backend.set_ingestion_status("proj", "ready", ctx=bound)
        before = await file_backend.get_state_version("proj", ctx=bound)
        assert before is not None

        await file_backend.delete_kb(COLLIDING)

        assert await file_backend.get_state_version("proj", ctx=bound) == before

    async def test_memory_backend_keeps_tenant_state(
        self, memory_backend: InMemoryKnowledgeBackend, bound: BoundTenantContext
    ) -> None:
        """Memory survived this by accident, and must keep surviving it.

        Its ``delete_kb`` matches state record keys by suffix rather than
        by prefix, so the colliding name never matched — the third
        divergent outcome from one call sequence. Pinned so the fix does
        not make the three backends agree by breaking this one.
        """
        await memory_backend.create_kb("proj")
        await memory_backend.create_kb(COLLIDING)
        await memory_backend.set_ingestion_status("proj", "ready", ctx=bound)
        before = await memory_backend.get_state_version("proj", ctx=bound)
        assert before is not None

        await memory_backend.delete_kb(COLLIDING)

        assert await memory_backend.get_state_version("proj", ctx=bound) == before

    async def test_tenant_state_does_not_block_creating_that_domain(
        self, file_backend: FileKnowledgeBackend, bound: BoundTenantContext
    ) -> None:
        """The milder face of the same overlap.

        With state written first, the colliding domain's directory already
        exists, so ``create_kb`` reported a knowledge base nobody had
        created as already existing.
        """
        await file_backend.create_kb("proj")
        await file_backend.set_ingestion_status("proj", "ready", ctx=bound)

        info = await file_backend.create_kb(COLLIDING)

        assert info.domain_id == COLLIDING

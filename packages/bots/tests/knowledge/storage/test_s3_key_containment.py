"""A resource path and a snapshot version stay inside their S3 slots.

The file backend bounded both of these a release ago; this backend
bounded neither, and the two consume the *same* identifiers through the
same public methods. That asymmetry is the finding: two backends that
were consistently unguarded became inconsistently guarded, so a
deployment's exposure depended on which one it configured.

**"S3 resolves nothing" is not a defence.** A ``..`` in a key is stored
literally, so nothing traverses inside S3 itself — and that is exactly
why it must be refused at the composing site rather than tolerated. The
bucket is routinely read by something that *does* resolve: ``aws s3
sync`` to a local tree, a CloudFront origin, this repository's own file
backend over the same layout. A key that only misbehaves once it is
copied somewhere is worse than one that misbehaves immediately, because
nothing on the write path reports it.

Nesting inside ``content/`` is the point of a content tree and is
untouched — this is containment, not the one-segment rule that governs
``domain_id`` (see ``test_domain_id_segment_rule``).

These tests compose keys and never open a client, so they need no
LocalStack: the defect is in the composition, and asserting on a live
``put_object`` would only prove that S3 stores whatever string it is
given, which it does.
"""

from __future__ import annotations

import pytest

from dataknobs_bots.knowledge.storage.key_layout import KnowledgeKeyKind
from dataknobs_bots.knowledge.storage.s3 import S3KnowledgeBackend
from dataknobs_common.paths import PathEscapeError, SegmentEscapeError
from dataknobs_common.tenancy import PrefixedTenantContext


@pytest.fixture
def backend() -> S3KnowledgeBackend:
    """A backend that never talks to S3 — only its key composition is used."""
    return S3KnowledgeBackend(bucket="test-bucket", prefix="kb/")


class TestAResourcePathStaysInsideTheContentTree:
    @pytest.mark.parametrize(
        "path",
        ["../_metadata.json", "../../other/content/x.md", "a/../../../escape.md"],
    )
    def test_a_path_that_climbs_out_is_refused(
        self, backend: S3KnowledgeBackend, path: str
    ) -> None:
        with pytest.raises(PathEscapeError, match="outside"):
            backend._s3_key("acme", path)

    def test_an_absolute_path_is_refused(self, backend: S3KnowledgeBackend) -> None:
        """The spelling that discards the base rather than climbing out."""
        with pytest.raises(PathEscapeError, match="outside"):
            backend._s3_key("acme", "/etc/passwd")

    def test_nesting_is_untouched(self, backend: S3KnowledgeBackend) -> None:
        assert backend._s3_key("acme", "docs/guide.md") == "kb/acme/content/docs/guide.md"

    def test_an_interior_parent_ref_that_stays_inside_is_collapsed(
        self, backend: S3KnowledgeBackend
    ) -> None:
        """Contained, and normalised to the key it actually addresses.

        S3 would otherwise store ``a/../b.md`` and ``b.md`` as two
        distinct objects for one intended file, so the same document
        would answer to one name and not the other depending on how the
        writer spelled it.
        """
        assert backend._s3_key("acme", "sub/../guide.md") == "kb/acme/content/guide.md"

    def test_the_domain_listing_prefix_is_unchanged(self, backend: S3KnowledgeBackend) -> None:
        """No path means the domain's prefix — what ``delete_kb`` paginates."""
        assert backend._s3_key("acme") == "kb/acme/"


class TestASnapshotVersionStaysInItsLineage:
    @pytest.mark.parametrize("version", ["../../../outside/secret", "a/b", "..", ""])
    def test_a_structured_version_is_refused(
        self, backend: S3KnowledgeBackend, version: str
    ) -> None:
        """``version`` is a caller-supplied token, not a computed digest.

        Every producer in this repo is an MD5 hex that cannot carry a
        separator — but the consumer is the caller, who persisted a
        token from ``list_changes_since`` and hands it back, and nothing
        on the way in constrains it to what was handed out.
        """
        with pytest.raises(SegmentEscapeError, match="snapshot version"):
            backend._snapshot_key("acme", version)

    def test_a_digest_shaped_version_still_composes(self, backend: S3KnowledgeBackend) -> None:
        digest = "12f7f01abb460de3d1f65d16d755b3f3"
        assert backend._snapshot_key("acme", digest) == (f"kb/acme/_snapshots/{digest}.json")


class TestTheTenantStatePrefixIsBounded:
    """The prefix is a consumer-supplied *pattern*, not an identifier.

    ``validate_tenant_id`` keeps a separator out of ``tenant_id``, and
    cannot see what the pattern itself contributes — which is why the
    file backend bounds the formatted result and this backend has to as
    well. Neither an id check upstream nor the segment rule reaches it:
    a prefix is several segments by design.
    """

    def test_a_pattern_that_climbs_out_is_refused(self, backend: S3KnowledgeBackend) -> None:
        ctx = PrefixedTenantContext(
            tenant_id="acme", domain_id="proj", prefix_pattern="../../{tenant_id}/"
        )
        with pytest.raises(PathEscapeError, match="tenant state prefix"):
            backend._metadata_key("proj", ctx)

    def test_a_non_canonical_pattern_is_refused_rather_than_rewritten(
        self, backend: S3KnowledgeBackend
    ) -> None:
        """Normalising it would move every state document for that tenant.

        ``a/../b/`` and ``b/`` address the same place once something
        resolves them and different places until then, so silently
        picking one is a data migration disguised as a fix.
        """
        ctx = PrefixedTenantContext(
            tenant_id="acme", domain_id="proj", prefix_pattern="tenants/../{tenant_id}/"
        )
        with pytest.raises(PathEscapeError, match="tenant state prefix"):
            backend._metadata_key("proj", ctx)

    def test_an_ordinary_prefix_composes_unchanged(self, backend: S3KnowledgeBackend) -> None:
        """The layout every reference context produces is untouched."""
        ctx = PrefixedTenantContext(
            tenant_id="acme", domain_id="proj", prefix_pattern="tenants/{tenant_id}/_state/"
        )
        assert backend._metadata_key("proj", ctx) == (
            "kb/_scoped/tenants/acme/_state/proj/_metadata.json"
        )

    def test_the_no_context_key_is_byte_identical(self, backend: S3KnowledgeBackend) -> None:
        """``ctx=None`` contributes no prefix, so nothing moved."""
        assert backend._metadata_key("proj") == "kb/proj/_metadata.json"

    @pytest.mark.parametrize("pattern", ["../../{tenant_id}/", "tenants/../{tenant_id}/"])
    def test_key_pattern_is_bounded_by_the_same_check(
        self, backend: S3KnowledgeBackend, pattern: str
    ) -> None:
        """The method that composed the raw prefix while its siblings did not.

        Every other helper on this class was routed through the bounded
        prefix; ``key_pattern`` kept calling the unbounded one, so a
        pattern that ``_metadata_key`` refused outright was returned here
        as a perfectly well-formed string naming a location no write will
        ever produce — an EventBridge rule or bucket notification that
        silently matches nothing, which is the failure mode that reports
        success.

        The bound now lives at the single point every composition goes
        through, rather than being something each new helper has to
        remember.
        """
        ctx = PrefixedTenantContext(tenant_id="acme", domain_id="proj", prefix_pattern=pattern)
        with pytest.raises(PathEscapeError, match="tenant state prefix"):
            backend.key_pattern(KnowledgeKeyKind.METADATA, "proj", ctx=ctx)

    def test_key_pattern_and_the_key_it_matches_share_a_root(
        self, backend: S3KnowledgeBackend
    ) -> None:
        """A pattern that does not cover its own key is the whole point."""
        ctx = PrefixedTenantContext(
            tenant_id="acme", domain_id="proj", prefix_pattern="tenants/{tenant_id}/_state/"
        )

        pattern = backend.key_pattern(KnowledgeKeyKind.METADATA, "proj", ctx=ctx)

        assert pattern == backend._metadata_key("proj", ctx)


class TestTheTwoBackendsNowAgree:
    """The asymmetry the finding names, asserted rather than described."""

    def test_both_backends_refuse_the_same_resource_path(
        self, backend: S3KnowledgeBackend, tmp_path
    ) -> None:
        from dataknobs_bots.knowledge.storage.file import FileKnowledgeBackend

        file_backend = FileKnowledgeBackend(base_path=tmp_path)
        with pytest.raises(PathEscapeError):
            file_backend._file_path("acme", "../../etc/passwd")
        with pytest.raises(PathEscapeError):
            backend._s3_key("acme", "../../etc/passwd")

    @pytest.mark.parametrize("version", ["../../../outside/secret", "a/b", "", ".."])
    def test_both_backends_refuse_the_same_snapshot_version(
        self, backend: S3KnowledgeBackend, tmp_path, version: str
    ) -> None:
        """``a/b`` is the case that showed the two rules were different.

        This assertion was made on ``../../../outside/secret`` alone, and
        that is the one input where containment and the segment rule
        happen to give the same answer — so the class asserting agreement
        asserted it on the only case that could not detect disagreement.
        ``a/b`` composes ``_snapshots/a/b.json``, which never leaves the
        snapshot directory: contained, so the file backend accepted it
        while S3 refused it. Both now ask the segment question, which is
        the right one for a name with one slot.
        """
        from dataknobs_bots.knowledge.storage.file import FileKnowledgeBackend

        file_backend = FileKnowledgeBackend(base_path=tmp_path)
        with pytest.raises(ValueError, match="snapshot version"):
            file_backend._snapshot_file("acme", version)
        with pytest.raises(ValueError, match="snapshot version"):
            backend._snapshot_key("acme", version)

    def test_both_backends_accept_the_digest_every_producer_emits(
        self, backend: S3KnowledgeBackend, tmp_path
    ) -> None:
        """The rule must not have been bought by breaking the feature."""
        from dataknobs_bots.knowledge.storage.file import FileKnowledgeBackend

        file_backend = FileKnowledgeBackend(base_path=tmp_path)
        digest = "12f7f01abb460de3d1f65d16d755b3f3"

        assert file_backend._snapshot_file("acme", digest).name == f"{digest}.json"
        assert backend._snapshot_key("acme", digest).endswith(f"{digest}.json")

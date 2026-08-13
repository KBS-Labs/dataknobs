"""A ``domain_id`` names one knowledge base, not a path through the layout.

Every backend interleaves ``domain_id`` with the layout's own literal
segments — ``content/``, ``_metadata.json``, ``_snapshots/`` — so a
separator inside it addresses a *different* knowledge base's slots:

    _metadata_key("acme/content")        -> {p}acme/content/_metadata.json
    _s3_key("acme", "_metadata.json")    -> {p}acme/content/_metadata.json

Those are the same object. The damage both directions is reproduced
below: an ordinary content file destroys another KB's metadata, and
``delete_kb`` on the shorter name deletes the longer one outright.

**Containment does not catch this**, which is why the guard shipped for
the file backend left it open: ``acme/content`` never leaves the base,
so ``safe_join`` is satisfied while the slot collides. The invariant
being asserted here is not "stays inside the tree" but "occupies one
segment of it".

The three backends disagreed about all of it before this rule — the same
call sequence overwrote on S3, refused at ``create_kb`` on the file
backend, and did nothing at all in memory — so the rule is asserted
against every backend rather than against the one that reported the
loudest failure. A ``domain_id`` refused in production must be refused in
the memory backend a consumer develops against.

Also pinned: a nested ``domain_id`` was **invisible to ``list_kbs()``**
on both persistent backends, which is the evidence that nesting was
never a supported spelling to preserve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataknobs_bots.knowledge.storage.file import FileKnowledgeBackend
from dataknobs_bots.knowledge.storage.memory import InMemoryKnowledgeBackend
from dataknobs_bots.knowledge.storage.s3 import S3KnowledgeBackend
from dataknobs_common.testing import requires_localstack

#: Spellings that do not name a single knowledge base. ``acme/content``
#: is the one that collides destructively; the rest are the same defect
#: reached by other spellings, including the absolute one that discards
#: the layout prefix on a filesystem backend.
NOT_A_SEGMENT = ["acme/content", "a/b", "..", ".", "", "   ", "/abs", "a\\b", "a\x00b"]

#: Spellings that must keep working. Nothing here reaches another KB's
#: slots, and refusing them would buy the guard by breaking the feature.
STILL_LEGAL = ["acme", "acme-content", "acme_content", "acme.v2", "ACME2"]


@pytest.fixture
async def file_backend(tmp_path: Path):
    backend = FileKnowledgeBackend(base_path=tmp_path / "kb")
    await backend.initialize()
    return backend


@pytest.fixture
async def memory_backend():
    backend = InMemoryKnowledgeBackend()
    await backend.initialize()
    return backend


class TestTheRuleHoldsOnEveryBackend:
    """Same refusal, same spellings, all three backends."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("domain_id", NOT_A_SEGMENT)
    async def test_file_backend_refuses(self, file_backend, domain_id: str) -> None:
        with pytest.raises(ValueError, match="domain_id"):
            await file_backend.create_kb(domain_id)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("domain_id", NOT_A_SEGMENT)
    async def test_memory_backend_refuses(self, memory_backend, domain_id: str) -> None:
        with pytest.raises(ValueError, match="domain_id"):
            await memory_backend.create_kb(domain_id)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("domain_id", STILL_LEGAL)
    async def test_a_plain_name_still_creates_on_file(self, file_backend, domain_id: str) -> None:
        info = await file_backend.create_kb(domain_id)
        assert info.domain_id == domain_id
        assert [kb.domain_id for kb in await file_backend.list_kbs()] == [domain_id]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("domain_id", STILL_LEGAL)
    async def test_a_plain_name_still_creates_in_memory(
        self, memory_backend, domain_id: str
    ) -> None:
        info = await memory_backend.create_kb(domain_id)
        assert info.domain_id == domain_id


class TestEveryDomainTakingEntryPointIsCovered:
    """A funnel missed by the guard is a live route to the same damage.

    Each of these composes a key or path from ``domain_id`` on some
    path through the class, so guarding only ``create_kb`` would leave
    the rest reachable.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda b: b.get_info("acme/content"), id="get_info"),
            pytest.param(lambda b: b.delete_kb("acme/content"), id="delete_kb"),
            pytest.param(lambda b: b.list_files("acme/content"), id="list_files"),
            pytest.param(lambda b: b.get_file("acme/content", "x.md"), id="get_file"),
            pytest.param(lambda b: b.file_exists("acme/content", "x.md"), id="file_exists"),
            pytest.param(lambda b: b.delete_file("acme/content", "x.md"), id="delete_file"),
            pytest.param(lambda b: b.put_file("acme/content", "x.md", b"x"), id="put_file"),
            pytest.param(lambda b: b.get_checksum("acme/content"), id="get_checksum"),
            pytest.param(lambda b: b.get_state_version("acme/content"), id="get_state_version"),
        ],
    )
    async def test_file_backend_entry_points(self, file_backend, call: Any) -> None:
        with pytest.raises(ValueError, match="domain_id"):
            await call(file_backend)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda b: b.get_info("acme/content"), id="get_info"),
            pytest.param(lambda b: b.delete_kb("acme/content"), id="delete_kb"),
            pytest.param(lambda b: b.list_files("acme/content"), id="list_files"),
            pytest.param(lambda b: b.get_file("acme/content", "x.md"), id="get_file"),
            pytest.param(lambda b: b.put_file("acme/content", "x.md", b"x"), id="put_file"),
            pytest.param(lambda b: b.get_state_version("acme/content"), id="get_state_version"),
        ],
    )
    async def test_memory_backend_entry_points(self, memory_backend, call: Any) -> None:
        with pytest.raises(ValueError, match="domain_id"):
            await call(memory_backend)

    @pytest.mark.asyncio
    async def test_key_pattern_refuses_a_nested_domain(self, file_backend) -> None:
        """The watch-pattern surface composes the same ingredients.

        It is the least severe member — the string is a glob handed to a
        subscription rather than a key anything writes — but a pattern
        built from a name that addresses another KB subscribes to that
        KB, which is the same defect one layer out.
        """
        with pytest.raises(ValueError, match="domain_id"):
            file_backend.key_pattern(domain_id="acme/content")

    @pytest.mark.asyncio
    async def test_key_pattern_still_accepts_the_all_domains_spelling(self, file_backend) -> None:
        """``domain_id=None`` means "every domain" and is not a name."""
        assert file_backend.key_pattern() != ""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("domain_id", ["acme/content", ""])
    async def test_memory_key_pattern_refuses_what_the_others_refuse(
        self, memory_backend, domain_id: str
    ) -> None:
        """It returns no pattern; that is not a reason to accept any name.

        Producing ``""`` is a property of in-process storage — no
        external observer can filter against it. Validation is a separate
        question, and skipping it made "an empty ``domain_id`` is refused
        rather than read as the all-domains wildcard" true of two
        backends out of three, on the one a consumer develops against.
        """
        with pytest.raises(ValueError, match="domain_id"):
            memory_backend.key_pattern(domain_id=domain_id)

    @pytest.mark.asyncio
    async def test_memory_key_pattern_still_accepts_the_all_domains_spelling(
        self, memory_backend
    ) -> None:
        assert memory_backend.key_pattern() == ""


class TestTheReproducedDamage:
    """The two failures that made this a security fix, not a tidy-up."""

    @pytest.mark.asyncio
    async def test_a_content_file_cannot_reach_another_kbs_metadata_slot(
        self, memory_backend
    ) -> None:
        """``put_file(acme, "_metadata.json")`` addressed KB ``acme/content``.

        On S3 this overwrote the victim's metadata with attacker-supplied
        JSON; on the file backend the same two keys collide. The route is
        closed by refusing the domain that names the collision, so the
        victim KB cannot be created to be overwritten.
        """
        await memory_backend.create_kb("acme")
        with pytest.raises(ValueError, match="domain_id"):
            await memory_backend.create_kb("acme/content")

    @pytest.mark.asyncio
    async def test_delete_kb_cannot_be_aimed_at_a_prefix_of_another_kb(self, file_backend) -> None:
        """``delete_kb("acme")`` deleted KB ``acme/content`` entirely.

        Both persistent backends delete by prefix — S3 paginates over
        ``{prefix}{domain}/`` and the file backend ``rmtree``s the
        directory — so a nested KB was collateral. With one-segment
        domains no KB is a prefix of another's *layout*, so a delete
        reaches only what it names.
        """
        await file_backend.create_kb("acme")
        await file_backend.create_kb("acmex")
        assert await file_backend.delete_kb("acme") is True
        assert [kb.domain_id for kb in await file_backend.list_kbs()] == ["acmex"]


@requires_localstack
@pytest.mark.integration
@pytest.mark.s3
class TestTheRuleHoldsAgainstRealS3:
    """The backend where the damage was executed end to end."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("domain_id", NOT_A_SEGMENT)
    async def test_s3_backend_refuses(self, s3_kb_config, domain_id: str) -> None:
        backend = S3KnowledgeBackend.from_config(s3_kb_config)
        await backend.initialize()
        try:
            with pytest.raises(ValueError, match="domain_id"):
                await backend.create_kb(domain_id)
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_the_colliding_pair_can_no_longer_both_exist(self, s3_kb_config) -> None:
        """The executed exploit, as a regression guard.

        Before: ``create_kb("acme")`` and ``create_kb("acme/content")``
        both succeeded, and then ``put_file("acme", "_metadata.json", …)``
        replaced the second KB's metadata document with whatever the
        first KB's writer supplied.
        """
        backend = S3KnowledgeBackend.from_config(s3_kb_config)
        await backend.initialize()
        try:
            await backend.create_kb("acme", {"owner": "acme-team"})
            with pytest.raises(ValueError, match="domain_id"):
                await backend.create_kb("acme/content", {"owner": "victim"})
            # The colliding key is still writable as an ordinary content
            # file — it is only a metadata slot for a domain that can no
            # longer exist.
            await backend.put_file("acme", "_metadata.json", b"just a file")
            info = await backend.get_info("acme")
            assert info is not None
            assert info.to_dict()["metadata"] == {"owner": "acme-team"}
        finally:
            await backend.close()

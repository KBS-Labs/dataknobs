"""A ``DocumentFileRef`` names a file inside the source's root.

:class:`LocalDocumentSource` declares a ``root`` and its own
``iter_files`` derives every ref it yields with ``relative_to(root)``,
so the class already behaves as though the root bounds it — but
``read_bytes`` and ``read_streaming`` composed ``root / ref.path`` and
opened the result, and a caller may construct a ref itself. The public
signature invites exactly that: :class:`DocumentFileRef` is an exported
dataclass, and the protocol's own delta-ingest seam hands refs back to
the source that did not necessarily come from it.

Both spellings are covered at both methods, because they fail
differently. A ``..`` segment climbs out of the root; an **absolute**
``path`` discards it outright, since ``Path("/root") / "/etc/passwd"``
is ``/etc/passwd`` — the wider of the two holes, not a narrower case.

Escape assertions match on the message rather than on a bare
``ValueError``: :class:`~dataknobs_common.paths.PathEscapeError` is a
``ValueError``, and so is a good deal else, so a bare assertion would
pass on an unrelated failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_common.paths import PathEscapeError
from dataknobs_xization.ingestion.source import DocumentFileRef, LocalDocumentSource


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A source root with a secret parked beside it, outside."""
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "a.md").write_text("# A\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secrets.env").write_text("API_KEY=hunter2\n")
    return root


def _ref(path: str) -> DocumentFileRef:
    return DocumentFileRef(path=path, size_bytes=-1, source_uri="test://ref")


async def _drain(source: LocalDocumentSource, ref: DocumentFileRef) -> bytes:
    return b"".join([chunk async for chunk in source.read_streaming(ref)])


class TestARefMayNotLeaveTheRoot:
    """The two escaping spellings, at both reading methods."""

    @pytest.mark.asyncio
    async def test_read_bytes_refuses_a_relative_escape(self, tree: Path) -> None:
        source = LocalDocumentSource(tree)
        with pytest.raises(PathEscapeError, match="outside"):
            await source.read_bytes(_ref("../outside/secrets.env"))

    @pytest.mark.asyncio
    async def test_read_streaming_refuses_a_relative_escape(self, tree: Path) -> None:
        source = LocalDocumentSource(tree)
        with pytest.raises(PathEscapeError, match="outside"):
            await _drain(source, _ref("../outside/secrets.env"))

    @pytest.mark.asyncio
    async def test_read_bytes_refuses_an_absolute_path(self, tree: Path) -> None:
        """An absolute ``path`` discards the root rather than climbing out.

        This is the spelling that makes an ``is_absolute()`` carve-out
        the wider hole rather than the narrower one — the whole root is
        gone, not merely escaped.
        """
        secret = tree.parent / "outside" / "secrets.env"
        source = LocalDocumentSource(tree)
        with pytest.raises(PathEscapeError, match="outside"):
            await source.read_bytes(_ref(str(secret)))

    @pytest.mark.asyncio
    async def test_read_streaming_refuses_an_absolute_path(self, tree: Path) -> None:
        secret = tree.parent / "outside" / "secrets.env"
        source = LocalDocumentSource(tree)
        with pytest.raises(PathEscapeError, match="outside"):
            await _drain(source, _ref(str(secret)))

    @pytest.mark.asyncio
    async def test_the_refusal_precedes_the_filesystem(self, tree: Path) -> None:
        """A name that escapes fails the same way whether or not it exists.

        Containment is lexical and runs before any read, so refusal
        cannot be used to probe for the existence of files outside the
        root.
        """
        source = LocalDocumentSource(tree)
        with pytest.raises(PathEscapeError, match="outside"):
            await source.read_bytes(_ref("../outside/does-not-exist.env"))


class TestContainedRefsStillRead:
    """The guard must not be bought by breaking ordinary nesting."""

    @pytest.mark.asyncio
    async def test_a_nested_ref_reads(self, tree: Path) -> None:
        source = LocalDocumentSource(tree)
        assert await source.read_bytes(_ref("sub/a.md")) == b"# A\n"

    @pytest.mark.asyncio
    async def test_an_interior_dotdot_that_stays_inside_reads(self, tree: Path) -> None:
        source = LocalDocumentSource(tree)
        assert await source.read_bytes(_ref("sub/../sub/a.md")) == b"# A\n"

    @pytest.mark.asyncio
    async def test_an_absolute_ref_inside_the_root_reads(self, tree: Path) -> None:
        """Containment is judged on where the ref lands, not how it is spelled."""
        source = LocalDocumentSource(tree)
        assert await source.read_bytes(_ref(str(tree / "sub" / "a.md"))) == b"# A\n"

    @pytest.mark.asyncio
    async def test_streaming_reads_a_nested_ref(self, tree: Path) -> None:
        source = LocalDocumentSource(tree)
        assert await _drain(source, _ref("sub/a.md")) == b"# A\n"

    @pytest.mark.asyncio
    async def test_every_ref_iter_files_yields_is_readable(self, tree: Path) -> None:
        """The source's own output must survive its own guard.

        ``iter_files`` derives each ref with ``relative_to(root)``, so
        this is the property that says the guard bounds callers rather
        than the class itself.
        """
        source = LocalDocumentSource(tree)
        read = [await source.read_bytes(ref) async for ref in source.iter_files(["**/*.md"])]
        assert read == [b"# A\n"]

"""A domain id or resource path must not address outside the backend's base.

:class:`FileKnowledgeBackend` turns two caller-supplied identifiers into
filesystem locations — ``domain_id`` via ``_kb_path`` and a resource
``path`` via ``_file_path`` — and both reach destructive sinks:
``mkdir(parents=True)``, ``shutil.rmtree``, ``unlink``, and an atomic
content write.

Each test asserts on the **sink**, not on the composed path: that the
call is refused *and* that nothing appeared, was read, or was removed
outside the base. A path assertion alone cannot tell "composed wrong"
from "composed wrong and acted on it", and it is the acting that
matters here — before the guard, ``delete_kb("../sacrificial")``
returned ``True`` having removed the tree.

Both escape spellings are covered per site, because a guard that catches
one is not a guard: a ``..`` segment, and an **absolute** part, which
discards the base outright (``Path("/base") / "/etc/passwd"`` is
``/etc/passwd``). A contained name that merely *looks* dangerous — a
subdirectory, or an interior ``a/../b`` — must still work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_bots.knowledge.storage.file import FileKnowledgeBackend


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """The backend's base directory, with a sibling it must never reach."""
    inside = tmp_path / "base"
    inside.mkdir()
    (tmp_path / "outside").mkdir()
    return inside


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    """A sibling of the base, standing in for anything else on the volume."""
    return tmp_path / "outside"


async def _backend(base: Path) -> FileKnowledgeBackend:
    backend = FileKnowledgeBackend(base_path=base)
    await backend.initialize()
    return backend


# --- domain_id -> _kb_path (S1) ------------------------------------------


async def test_create_kb_refuses_a_domain_id_that_walks_out(base: Path, outside: Path) -> None:
    """``mkdir(parents=True)`` must not run outside the base."""
    backend = await _backend(base)

    with pytest.raises(ValueError):
        await backend.create_kb("../outside/pwned")

    assert not (outside / "pwned").exists()


async def test_create_kb_refuses_an_absolute_domain_id(base: Path, outside: Path) -> None:
    """An absolute part discards the base; rejecting ``..`` alone misses it."""
    backend = await _backend(base)
    target = outside / "pwned-absolute"

    with pytest.raises(ValueError):
        await backend.create_kb(str(target))

    assert not target.exists()


async def test_delete_kb_refuses_a_domain_id_that_walks_out(base: Path, tmp_path: Path) -> None:
    """``shutil.rmtree`` is the sharpest sink in the class."""
    backend = await _backend(base)
    sacrificial = tmp_path / "outside" / "sacrificial"
    sacrificial.mkdir()
    (sacrificial / "precious.txt").write_text("keep me")

    with pytest.raises(ValueError):
        await backend.delete_kb("../outside/sacrificial")

    assert (sacrificial / "precious.txt").read_text() == "keep me"


async def test_get_info_refuses_a_domain_id_that_walks_out(base: Path) -> None:
    """A read-shaped method raises rather than answering about outside."""
    backend = await _backend(base)

    with pytest.raises(ValueError):
        await backend.get_info("../outside")


async def test_a_domain_id_naming_a_subdirectory_still_works(base: Path) -> None:
    """Containment is not a ``/``-rejecting character class."""
    backend = await _backend(base)

    info = await backend.create_kb("team/alpha")

    assert info.domain_id == "team/alpha"
    assert (base / "team" / "alpha").is_dir()


async def test_a_domain_id_with_an_interior_parent_ref_still_works(base: Path) -> None:
    """``a/../b`` never leaves the base, so it is contained."""
    backend = await _backend(base)

    await backend.create_kb("team/../beta")

    assert (base / "beta").is_dir()


# --- resource path -> _file_path (S2) ------------------------------------


async def test_put_file_refuses_a_path_that_walks_out(base: Path, outside: Path) -> None:
    """The atomic content write must not land outside the base."""
    backend = await _backend(base)
    await backend.create_kb("dom")

    with pytest.raises(ValueError):
        await backend.put_file("dom", "../../../outside/pwned.md", b"owned")

    assert not (outside / "pwned.md").exists()


async def test_put_file_refuses_an_absolute_path(base: Path, outside: Path) -> None:
    backend = await _backend(base)
    await backend.create_kb("dom")
    target = outside / "pwned-absolute.md"

    with pytest.raises(ValueError):
        await backend.put_file("dom", str(target), b"owned")

    assert not target.exists()


async def test_get_file_refuses_to_read_outside_the_base(base: Path, outside: Path) -> None:
    """Before the guard this returned the file's bytes, not ``None``."""
    backend = await _backend(base)
    await backend.create_kb("dom")
    (outside / "secret.txt").write_text("SECRET")

    with pytest.raises(ValueError):
        await backend.get_file("dom", "../../../outside/secret.txt")


async def test_delete_file_refuses_to_unlink_outside_the_base(base: Path, outside: Path) -> None:
    backend = await _backend(base)
    await backend.create_kb("dom")
    victim = outside / "victim.txt"
    victim.write_text("keep me")

    with pytest.raises(ValueError):
        await backend.delete_file("dom", "../../../outside/victim.txt")

    assert victim.read_text() == "keep me"


async def test_file_exists_refuses_a_path_that_walks_out(base: Path, outside: Path) -> None:
    """An escaping name is never a legitimate "absent" answer."""
    backend = await _backend(base)
    await backend.create_kb("dom")
    (outside / "probe.txt").write_text("x")

    with pytest.raises(ValueError):
        await backend.file_exists("dom", "../../../outside/probe.txt")


async def test_stream_file_refuses_a_path_that_walks_out(base: Path, outside: Path) -> None:
    backend = await _backend(base)
    await backend.create_kb("dom")
    (outside / "streamed.txt").write_text("SECRET")

    with pytest.raises(ValueError):
        await backend.stream_file("dom", "../../../outside/streamed.txt")


async def test_a_resource_path_in_a_subdirectory_still_works(base: Path) -> None:
    """The content tree is explicitly nested; ``subdir/file`` is normal."""
    backend = await _backend(base)
    await backend.create_kb("dom")

    await backend.put_file("dom", "subdir/nested.md", b"fine")

    assert await backend.get_file("dom", "subdir/nested.md") == b"fine"

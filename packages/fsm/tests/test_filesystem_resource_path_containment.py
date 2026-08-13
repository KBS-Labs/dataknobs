"""A path handed to ``FileSystemResource`` must not leave its base.

``FileSystemResource.__init__`` does ``Path(base_path).resolve()``, which
is only meaningful if that directory is a boundary — but nothing checked
a composed path against it. Four methods each did ``self.base_path /
path`` independently and acted on the result: ``open`` in read *or*
write mode, ``exists``, ``unlink``, and a ``glob``.

Each test asserts on the **sink** — that the file outside is not read,
not written, not removed — rather than on the composed path, because it
is the acting that matters. Both escape spellings are covered per site:
a ``..`` segment, and an **absolute** path, which discards the base
outright (``Path("/base") / "/etc/passwd"`` is ``/etc/passwd``), so a
guard that rejects only ``..`` is not a guard.

``delete`` gets its own attention. Its body is wrapped in
``except Exception: return False``, and ``False`` is also its ordinary
"the file was not there" answer — so a refusal raised *inside* that
``try`` would be swallowed into a result the caller cannot distinguish
from a no-op. The guard therefore runs before it, and
``test_delete_refuses_visibly_rather_than_returning_false`` is what
pins that ordering.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dataknobs_common.paths import PathEscapeError

from dataknobs_fsm.resources.filesystem import FileSystemResource


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """The resource's base directory."""
    inside = tmp_path / "base"
    inside.mkdir()
    return inside


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    """A sibling of the base, holding something worth protecting."""
    sibling = tmp_path / "outside"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("SECRET")
    return sibling


@pytest.fixture
def resource(base: Path) -> FileSystemResource:
    return FileSystemResource(name="fs", base_path=str(base))


# --- acquire / open -------------------------------------------------------


def test_open_refuses_to_read_outside_the_base(resource: FileSystemResource, outside: Path) -> None:
    """Before the guard this returned the file's contents."""
    with pytest.raises(PathEscapeError):
        with resource.open("../outside/secret.txt") as handle:
            handle.read()


def test_open_refuses_an_absolute_path(resource: FileSystemResource, outside: Path) -> None:
    with pytest.raises(PathEscapeError):
        with resource.open(str(outside / "secret.txt")) as handle:
            handle.read()


def test_open_for_write_refuses_to_create_outside_the_base(
    resource: FileSystemResource, outside: Path
) -> None:
    """Write mode also runs ``parent.mkdir(parents=True)``, so an escaping
    path built directories outside the base before writing into them.
    """
    with pytest.raises(PathEscapeError):
        with resource.open("../outside/planted/pwned.txt", "w") as handle:
            handle.write("owned")

    assert not (outside / "planted").exists()


# --- exists ---------------------------------------------------------------


def test_exists_refuses_rather_than_answering_about_outside(
    resource: FileSystemResource, outside: Path
) -> None:
    """Reporting True/False about a file outside the base is still
    reporting about it — an existence oracle over the whole volume.
    """
    with pytest.raises(PathEscapeError):
        resource.exists("../outside/secret.txt")


# --- delete ---------------------------------------------------------------


def test_delete_refuses_to_unlink_outside_the_base(
    resource: FileSystemResource, outside: Path
) -> None:
    victim = outside / "secret.txt"

    with pytest.raises(PathEscapeError):
        resource.delete("../outside/secret.txt")

    assert victim.read_text() == "SECRET"


def test_delete_refuses_visibly_rather_than_returning_false(
    resource: FileSystemResource, outside: Path
) -> None:
    """The guard must sit outside ``delete``'s blanket ``except``.

    ``delete`` answers ``False`` for "no such file". If the containment
    check ran inside the ``try``, the ``except Exception: return False``
    would turn a refusal into that same answer, and a caller could not
    tell a refused traversal from an ordinary miss. This asserts the
    refusal is raised, not folded into the return value.
    """
    absent = resource.delete("not-there.txt")
    assert absent is False  # the ordinary miss

    with pytest.raises(PathEscapeError):
        resource.delete("../outside/secret.txt")


# --- list_files -----------------------------------------------------------


def test_list_files_refuses_a_pattern_that_climbs_out(
    resource: FileSystemResource, outside: Path
) -> None:
    """A glob pattern composes onto the base exactly like a path does."""
    with pytest.raises(PathEscapeError):
        resource.list_files("../outside/*")


# --- the base's own legitimate shapes still work -------------------------


def test_a_path_in_a_subdirectory_still_works(resource: FileSystemResource, base: Path) -> None:
    """Nesting is the point of a file resource; only leaving is refused."""
    with resource.open("sub/nested.txt", "w") as handle:
        handle.write("fine")

    assert (base / "sub" / "nested.txt").read_text() == "fine"
    assert resource.exists("sub/nested.txt")
    assert resource.list_files("sub/*") == ["sub/nested.txt"]
    assert resource.delete("sub/nested.txt") is True


def test_an_interior_parent_ref_that_stays_inside_still_works(
    resource: FileSystemResource, base: Path
) -> None:
    """``a/../b`` never leaves the base, so it is contained."""
    with resource.open("sub/../top.txt", "w") as handle:
        handle.write("fine")

    assert (base / "top.txt").read_text() == "fine"

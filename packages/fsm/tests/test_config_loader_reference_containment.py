"""`$include` / `$import` resolve inside the config tree, and only inside it.

Two things are pinned here, and the first is the reason the second could go
unnoticed: **the reference feature itself had no tests at all**. `$include` and
`$import` appear nowhere in this package's suite and in no example config in the
repository, so nothing established what they do before anything could establish
what they must not do.

The containment half follows the reproduce-first order — every escape below was
executed against the unguarded loader first and read a file from outside the
tree. The behavioural half is what says the guard did not buy containment by
breaking the feature.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from dataknobs_common.config_loading import ConfigLoadError
from dataknobs_common.paths import PathEscapeError

from dataknobs_fsm.config.loader import ConfigLoader


def _fsm(name: str, network: str = "main") -> dict:
    """A minimal valid FSM config, tagged so its origin file is identifiable."""
    return {
        "name": name,
        "main_network": network,
        "networks": [
            {
                "name": network,
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "end", "is_end": True},
                ],
                "arcs": [{"from": "start", "to": "end", "name": "finish"}],
            }
        ],
    }


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload))
    return path


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A config tree with a fragment beside it and a secret outside it.

    <tmp>/configs/main.yaml       entry file — the root is its parent
    <tmp>/configs/shared.yaml     a fragment inside the tree
    <tmp>/configs/sub/            a subdirectory the tree descends into
    <tmp>/outside/stolen.yaml     NOT in the tree
    """
    _write(tmp_path / "configs" / "shared.yaml", _fsm("from-shared"))
    _write(tmp_path / "outside" / "stolen.yaml", _fsm("STOLEN"))
    (tmp_path / "configs" / "sub").mkdir(parents=True, exist_ok=True)
    return tmp_path


# ---------------------------------------------------------------------------
# The feature, which nothing covered before this file
# ---------------------------------------------------------------------------


def test_include_merges_a_fragment_beside_the_entry_file(tree: Path) -> None:
    entry = _write(tree / "configs" / "main.yaml", {"$include": "shared.yaml"})

    config = ConfigLoader().load_from_file(entry)

    assert config.name == "from-shared"


def test_include_resolves_relative_to_the_file_that_wrote_it(tree: Path) -> None:
    """A nested `$include` is spelled relative to its own file, not to the root.

    This is `dataknobs-fsm`'s spelling and it differs from
    `dataknobs-config`'s root-relative `extends:`. The difference is
    pre-existing semantics: bounding the tree must not change it.
    """
    _write(tree / "configs" / "sub" / "leaf.yaml", _fsm("from-leaf"))
    _write(tree / "configs" / "sub" / "frag.yaml", {"$include": "leaf.yaml"})
    entry = _write(tree / "configs" / "main.yaml", {"$include": "sub/frag.yaml"})

    config = ConfigLoader().load_from_file(entry)

    assert config.name == "from-leaf"


def test_a_fragment_in_a_subdirectory_may_reach_a_sibling_inside_the_tree(
    tree: Path,
) -> None:
    """The case that decides root-anchoring over per-hop bounding.

    ``configs/sub/frag.yaml`` naming ``../shared.yaml`` addresses
    ``configs/shared.yaml`` — plainly inside the config tree, and the ordinary
    shape of a shared-fragment directory. A guard bounding each hop to *its
    own* parent rejects this while contained; that is the consumer break the
    root anchor exists to avoid, so it is pinned rather than left to review.
    """
    _write(tree / "configs" / "sub" / "frag.yaml", {"$include": "../shared.yaml"})
    entry = _write(tree / "configs" / "main.yaml", {"$include": "sub/frag.yaml"})

    config = ConfigLoader().load_from_file(entry)

    assert config.name == "from-shared"


def test_import_pulls_one_nested_path_out_of_a_fragment(tree: Path) -> None:
    _write(
        tree / "configs" / "library.yaml",
        {"unused": {"name": "wrong"}, "flows": {"checkout": _fsm("from-import")}},
    )
    entry = _write(
        tree / "configs" / "main.yaml",
        {"$import": {"file": "library.yaml", "path": "flows.checkout"}},
    )

    config = ConfigLoader().load_from_file(entry)

    assert config.name == "from-import"


# ---------------------------------------------------------------------------
# Containment — each of these read `outside/stolen.yaml` before the guard
# ---------------------------------------------------------------------------


def test_include_may_not_climb_out_of_the_config_tree(tree: Path) -> None:
    entry = _write(tree / "configs" / "main.yaml", {"$include": "../outside/stolen.yaml"})

    with pytest.raises(PathEscapeError, match="outside"):
        ConfigLoader().load_from_file(entry)


def test_include_may_not_name_an_absolute_path(tree: Path) -> None:
    """The spelling a `..`-only guard misses.

    ``Path("/tree") / "/outside/stolen.yaml"`` is ``/outside/stolen.yaml`` —
    the base is not climbed out of, it is discarded.
    """
    stolen = tree / "outside" / "stolen.yaml"
    entry = _write(tree / "configs" / "main.yaml", {"$include": str(stolen)})

    with pytest.raises(PathEscapeError, match="outside"):
        ConfigLoader().load_from_file(entry)


def test_a_nested_include_may_not_climb_out_of_the_tree_it_started_in(
    tree: Path,
) -> None:
    """Containment is judged against the root, at every depth.

    The escape is written from a *fragment*, one hop below the entry file, so
    the value that escapes is never seen by the entry file's own directory.
    """
    _write(
        tree / "configs" / "sub" / "frag.yaml",
        {"$include": "../../outside/stolen.yaml"},
    )
    entry = _write(tree / "configs" / "main.yaml", {"$include": "sub/frag.yaml"})

    with pytest.raises(PathEscapeError, match="outside"):
        ConfigLoader().load_from_file(entry)


def test_import_may_not_climb_out_of_the_config_tree(tree: Path) -> None:
    """`$import` is a second composing site, not a variant spelling of the first."""
    _write(tree / "outside" / "library.yaml", {"flows": {"checkout": _fsm("STOLEN")}})
    entry = _write(
        tree / "configs" / "main.yaml",
        {"$import": {"file": "../outside/library.yaml", "path": "flows.checkout"}},
    )

    with pytest.raises(PathEscapeError, match="outside"):
        ConfigLoader().load_from_file(entry)


def test_import_may_not_name_an_absolute_path(tree: Path) -> None:
    library = _write(tree / "outside" / "library.yaml", {"flows": {"checkout": _fsm("STOLEN")}})
    entry = _write(
        tree / "configs" / "main.yaml",
        {"$import": {"file": str(library), "path": "flows.checkout"}},
    )

    with pytest.raises(PathEscapeError, match="outside"):
        ConfigLoader().load_from_file(entry)


def test_the_refusal_names_the_value_the_config_supplied(tree: Path) -> None:
    """A refusal has to be actionable: it quotes the reference, not the join."""
    entry = _write(tree / "configs" / "main.yaml", {"$include": "../outside/stolen.yaml"})

    with pytest.raises(PathEscapeError) as excinfo:
        ConfigLoader().load_from_file(entry)

    assert "../outside/stolen.yaml" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The anchor is the entry file's directory by default, and settable
# ---------------------------------------------------------------------------


def test_a_wider_config_root_admits_a_shared_directory_beside_the_tree(
    tree: Path,
) -> None:
    """The migration for a layout that deliberately spans sibling directories.

    Widening the anchor is the honest expression of "this reference is legal",
    and it stays a boundary: `shared/` is now inside the tree, `outside/` still
    is not. That is strictly more useful than an on/off escape hatch, which
    would admit both.
    """
    _write(tree / "app" / "fsm" / "flow.yaml", {"$include": "../shared/common.yaml"})
    _write(tree / "app" / "shared" / "common.yaml", _fsm("from-shared-tree"))
    entry = tree / "app" / "fsm" / "flow.yaml"

    with pytest.raises(PathEscapeError):
        ConfigLoader().load_from_file(entry)

    config = ConfigLoader().load_from_file(entry, config_root=tree / "app")
    assert config.name == "from-shared-tree"


def test_a_wider_config_root_is_still_a_boundary(tree: Path) -> None:
    _write(tree / "app" / "fsm" / "flow.yaml", {"$include": "../../outside/stolen.yaml"})
    entry = tree / "app" / "fsm" / "flow.yaml"

    with pytest.raises(PathEscapeError, match="outside"):
        ConfigLoader().load_from_file(entry, config_root=tree / "app")


def test_a_config_root_that_does_not_contain_the_entry_file_is_refused(
    tree: Path,
) -> None:
    """The two arguments disagree about which tree is being loaded."""
    entry = _write(tree / "configs" / "main.yaml", {"$include": "shared.yaml"})

    with pytest.raises(PathEscapeError, match="entry file"):
        ConfigLoader().load_from_file(entry, config_root=tree / "outside")


def test_a_widened_root_still_resolves_references_relative_to_their_own_file(
    tree: Path,
) -> None:
    """The entry file's own references do not become root-relative.

    With ``config_root`` widened, ``app/fsm/flow.yaml`` sits below the root.
    A bare ``common.yaml`` in it must still mean *beside flow.yaml* — reading
    it as ``app/common.yaml`` would silently load a different file rather than
    refuse, which is the failure mode a starting ``rel_base`` of ``""`` has.
    """
    _write(tree / "app" / "common.yaml", _fsm("WRONG-at-root"))
    _write(tree / "app" / "fsm" / "common.yaml", _fsm("right-beside-the-entry"))
    entry = _write(tree / "app" / "fsm" / "flow.yaml", {"$include": "common.yaml"})

    config = ConfigLoader().load_from_file(entry, config_root=tree / "app")

    assert config.name == "right-beside-the-entry"


def test_an_absolute_include_landing_inside_the_tree_still_loads(tree: Path) -> None:
    """Containment judges where a reference lands, not how it is spelled.

    The refusal of an absolute reference is not a rule against the spelling —
    it is the same containment question asked of a path that discards the
    base. One pointing back into the tree is inside it, so it loads, and
    pinning that here is what stops the guard from being reintroduced as an
    ``is_absolute()`` rejection.
    """
    entry = _write(
        tree / "configs" / "main.yaml",
        {"$include": str(tree / "configs" / "shared.yaml")},
    )

    config = ConfigLoader().load_from_file(entry)

    assert config.name == "from-shared"


# ---------------------------------------------------------------------------
# Malformed references — the feature's other untested edges
# ---------------------------------------------------------------------------


def test_a_self_referential_include_is_refused(tree: Path) -> None:
    """A file including itself recursed until the interpreter gave out.

    `RecursionError` is not catchable as a config problem, carries a
    thousand-frame traceback, and names no file — so a one-line typo in a
    fragment surfaced as an apparent interpreter fault.
    """
    entry = _write(tree / "configs" / "main.yaml", {"$include": "main.yaml"})

    with pytest.raises(ConfigLoadError, match="cycle"):
        ConfigLoader().load_from_file(entry)


def test_a_cycle_between_two_fragments_is_refused(tree: Path) -> None:
    """The same defect one hop out, where a reader is far less likely to see it."""
    _write(tree / "configs" / "a.yaml", {"$include": "b.yaml"})
    _write(tree / "configs" / "b.yaml", {"$include": "a.yaml"})
    entry = _write(tree / "configs" / "main.yaml", {"$include": "a.yaml"})

    with pytest.raises(ConfigLoadError, match="cycle"):
        ConfigLoader().load_from_file(entry)


def test_the_cycle_refusal_names_the_files_involved(tree: Path) -> None:
    _write(tree / "configs" / "a.yaml", {"$include": "a.yaml"})
    entry = _write(tree / "configs" / "main.yaml", {"$include": "a.yaml"})

    with pytest.raises(ConfigLoadError) as excinfo:
        ConfigLoader().load_from_file(entry)

    assert "a.yaml" in str(excinfo.value)


def test_the_same_fragment_may_be_included_twice_without_being_a_cycle(
    tree: Path,
) -> None:
    """Re-inclusion is not recursion, and the guard must not confuse them.

    A shared fragment pulled in by two siblings is the ordinary reason the
    include cache exists; a cycle check keyed on "seen before" rather than on
    "currently open" would refuse it.
    """
    _write(tree / "configs" / "sub" / "one.yaml", {"$include": "../shared.yaml"})
    _write(tree / "configs" / "sub" / "two.yaml", {"$include": "../shared.yaml"})
    entry = _write(
        tree / "configs" / "main.yaml",
        {"$include": "sub/one.yaml", "metadata": {"$include": "sub/two.yaml"}},
    )

    config = ConfigLoader().load_from_file(entry)

    assert config.name == "from-shared"
    assert config.metadata["name"] == "from-shared"


def test_a_fragment_changed_between_loads_is_read_again(tree: Path) -> None:
    """The include cache spans one load, not the loader's lifetime.

    `_included_configs` was never cleared, so a `ConfigLoader` kept across
    loads — the ordinary way to hold one — served the first load's copy of
    every fragment forever. Cheap within a load; wrong between them.
    """
    loader = ConfigLoader()
    entry = _write(tree / "configs" / "main.yaml", {"$include": "shared.yaml"})

    assert loader.load_from_file(entry).name == "from-shared"

    _write(tree / "configs" / "shared.yaml", _fsm("edited-on-disk"))

    assert loader.load_from_file(entry).name == "edited-on-disk"


def test_an_import_without_a_file_key_is_refused(tree: Path) -> None:
    """`value["file"]` raised a bare `KeyError` naming only `'file'`."""
    entry = _write(tree / "configs" / "main.yaml", {"$import": {"path": "networks"}})

    with pytest.raises(ConfigLoadError, match="file"):
        ConfigLoader().load_from_file(entry)


def test_an_import_whose_file_is_not_a_string_is_refused(tree: Path) -> None:
    entry = _write(tree / "configs" / "main.yaml", {"$import": {"file": ["a", "b"]}})

    with pytest.raises(ConfigLoadError, match="file"):
        ConfigLoader().load_from_file(entry)


def test_references_may_be_left_unresolved(tree: Path) -> None:
    """`resolve_references=False` means no composition happens, so no refusal.

    Pinned because the guard must live at the composition, not at the parse:
    a caller that has switched the feature off is not asking about containment.
    """
    entry = _write(
        tree / "configs" / "main.yaml",
        dict(_fsm("unresolved"), **{"$include": "../outside/stolen.yaml"}),
    )

    config = ConfigLoader().load_from_file(entry, resolve_references=False)

    assert config.name == "unresolved"

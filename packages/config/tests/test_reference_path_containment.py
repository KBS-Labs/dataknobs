"""An ``@``-reference is bounded by ``config_root``, in both spellings.

Any string in a config list that begins with ``@`` is read as a file
reference: ``Config._load_referenced_file`` composes it onto ``config_root``
and reads it. The value comes out of a config *file*, not from a caller — the
same provenance as ``extends:``, which ``find_config_file`` already bounds
including its absolute spelling.

Both branches escaped before this file, and each one loaded a file carrying a
secret from outside the config root.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from dataknobs_common.paths import PathEscapeError

from dataknobs_config import Config
from dataknobs_config.exceptions import ConfigError

# Matches the containment failure specifically, so an unrelated `ValueError`
# from a malformed config cannot report this green.
_ESCAPE = "outside"


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload))
    return path


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A config root, a fragment inside it, and a secret outside it."""
    _write(tmp_path / "configs" / "fragment.yaml", {"name": "from-fragment"})
    _write(tmp_path / "outside" / "secret.yaml", {"name": "stolen", "api_key": "SECRET"})
    return tmp_path


def _entry(tree: Path, reference: str) -> Path:
    return _write(tree / "configs" / "main.yaml", {"database": [f"@{reference}"]})


def test_a_relative_reference_inside_the_root_still_loads(tree: Path) -> None:
    entry = _entry(tree, "fragment.yaml")

    config = Config(entry)

    assert config.get("database")["name"] == "from-fragment"


def test_a_reference_into_a_subdirectory_still_loads(tree: Path) -> None:
    """Descending is legal; the reference just may not leave."""
    _write(tree / "configs" / "sub" / "nested.yaml", {"name": "from-nested"})
    entry = _entry(tree, "sub/nested.yaml")

    config = Config(entry)

    assert config.get("database")["name"] == "from-nested"


def test_a_relative_reference_may_not_climb_out_of_the_config_root(tree: Path) -> None:
    entry = _entry(tree, "../outside/secret.yaml")

    with pytest.raises(PathEscapeError, match=_ESCAPE):
        Config(entry)


def test_an_absolute_reference_may_not_discard_the_config_root(tree: Path) -> None:
    """The branch that never consulted ``config_root`` at all.

    ``is_absolute()`` short-circuited the composition, so this is not a
    narrower case than a ``..`` — it is the wider one. The provenance is
    identical in both spellings, and a name whose provenance is the config
    tree does not get to address outside it because it was written absolutely.
    """
    entry = _entry(tree, str(tree / "outside" / "secret.yaml"))

    with pytest.raises(PathEscapeError, match=_ESCAPE):
        Config(entry)


def test_the_refusal_names_the_reference_the_config_supplied(tree: Path) -> None:
    entry = _entry(tree, "../outside/secret.yaml")

    with pytest.raises(PathEscapeError) as excinfo:
        Config(entry)

    assert "../outside/secret.yaml" in str(excinfo.value)


class TestTheOptOutIsNotReachableFromConfigContent:
    """The guard exists because config content is the lower-trust plane.

    That is the whole premise: a reference is bounded *because* it comes out
    of a config file rather than from a caller. An off-switch readable from
    the same file leaves the guard bounding an input that can switch it off,
    which is not a boundary — so `settings:` refuses the key outright rather
    than honouring or silently dropping it.
    """

    def test_a_config_file_may_not_switch_off_its_own_guard(self, tree: Path) -> None:
        entry = _write(
            tree / "configs" / "main.yaml",
            {
                "settings": {"allow_reference_outside_config_root": True},
                "database": ["@../outside/secret.yaml"],
            },
        )

        with pytest.raises(ConfigError, match="allow_reference_outside_config_root"):
            Config(entry)

    def test_the_refusal_says_where_the_opt_out_does_live(self, tree: Path) -> None:
        """A silent drop fails closed and leaves the operator with no thread."""
        entry = _write(
            tree / "configs" / "main.yaml",
            {"settings": {"allow_reference_outside_config_root": True}},
        )

        with pytest.raises(ConfigError) as excinfo:
            Config(entry)

        assert "Config(" in str(excinfo.value)

    def test_a_dict_source_may_not_switch_it_off_either(self, tree: Path) -> None:
        """A parsed dict is the same plane as the file it was parsed from."""
        with pytest.raises(ConfigError, match="allow_reference_outside_config_root"):
            Config({"settings": {"allow_reference_outside_config_root": True}})

    def test_an_unsubstituted_placeholder_cannot_switch_the_guard_off(self, tree: Path) -> None:
        """The sharpest spelling of the same defect, and the least visible.

        `substitute_env_vars` runs on atomic configs only, after settings are
        taken raw — so a placeholder in `settings:` stays the literal string
        `"${...}"`. Read as a bare truthy value it disabled the guard
        permanently, whatever the variable was set to and including unset,
        for an operator doing the ordinary thing of templating a flag from
        the environment.
        """
        entry = _write(
            tree / "configs" / "main.yaml",
            {
                "settings": {"allow_reference_outside_config_root": "${DK_ALLOW_OUTSIDE}"},
                "database": ["@../outside/secret.yaml"],
            },
        )

        with pytest.raises(ConfigError, match="allow_reference_outside_config_root"):
            Config(entry)


class TestTheCallerSuppliedOptOut:
    """`allow_reference_outside_config_root=` — off by default, caller-only.

    `dataknobs-config` has the shared-directory case a file resource does not:
    `configs/app.yaml` with `extends: ../shared/base` is the documented layout
    this package already supports, so a deployment that references across
    trees on purpose needs a migration that is not "restructure your files".

    It is a constructor argument, which is where every sibling opt-out in this
    package already sits — `find_config_file(allow_outside=)`,
    `InheritableConfigLoader(allow_outside=)`, `EnvironmentConfig.load(allow_outside=)`.
    """

    def test_the_opt_out_admits_a_reference_outside_the_root(self, tree: Path) -> None:
        entry = _entry(tree, "../outside/secret.yaml")

        config = Config(entry, allow_reference_outside_config_root=True)

        assert config.get("database")["name"] == "stolen"

    def test_the_opt_out_admits_an_absolute_reference(self, tree: Path) -> None:
        entry = _entry(tree, str(tree / "outside" / "secret.yaml"))

        config = Config(entry, allow_reference_outside_config_root=True)

        assert config.get("database")["name"] == "stolen"

    def test_it_is_off_by_default(self, tree: Path) -> None:
        entry = _entry(tree, "../outside/secret.yaml")

        with pytest.raises(PathEscapeError, match=_ESCAPE):
            Config(entry)

    def test_from_file_takes_it_too(self, tree: Path) -> None:
        """The classmethod is the documented entry point for a file load."""
        entry = _entry(tree, "../outside/secret.yaml")

        config = Config.from_file(entry, allow_reference_outside_config_root=True)

        assert config.get("database")["name"] == "stolen"

    def test_the_opt_out_logs_only_when_a_reference_actually_escapes(
        self, tree: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`find_config_file`'s convention: the warning marks a real escape.

        Warning on every load while the flag is on trains the reader to
        ignore it, which is the same as not warning at all.
        """
        entry = _entry(tree, "fragment.yaml")

        with caplog.at_level("WARNING"):
            Config(entry, allow_reference_outside_config_root=True)

        assert not [r for r in caplog.records if "outside" in r.getMessage()]

    def test_an_escape_under_the_opt_out_is_logged(
        self, tree: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        entry = _entry(tree, "../outside/secret.yaml")

        with caplog.at_level("WARNING"):
            Config(entry, allow_reference_outside_config_root=True)

        assert [r for r in caplog.records if "outside" in r.getMessage()]


class TestTheRootItselfIsNotWidenableFromContent:
    """The same plane argument, for the boundary rather than the switch.

    `config_root` is a settings key, so a config file naming its own root
    would widen the tree it is bounded to — the off-switch again, spelled as
    a boundary. A file load pins the root to the entry file's own directory
    before the file's `settings:` block is read, so it cannot.
    """

    def test_an_entry_file_may_not_name_its_own_root(self, tree: Path) -> None:
        entry = _write(
            tree / "configs" / "main.yaml",
            {
                "settings": {"config_root": str(tree)},
                "database": ["@../outside/secret.yaml"],
            },
        )

        with pytest.raises(PathEscapeError, match=_ESCAPE):
            Config(entry)

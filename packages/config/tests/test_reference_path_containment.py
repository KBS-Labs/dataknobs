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


class TestTheDocumentedOptOut:
    """`allow_reference_outside_config_root` — off by default, per-deployment.

    `dataknobs-config` has the shared-directory case a file resource does not:
    `configs/app.yaml` with `extends: ../shared/base` is the documented layout
    this package already supports, so a deployment that references across
    trees on purpose needs a migration that is not "restructure your files".
    """

    def test_the_opt_out_admits_a_reference_outside_the_root(self, tree: Path) -> None:
        entry = _write(
            tree / "configs" / "main.yaml",
            {
                "settings": {"allow_reference_outside_config_root": True},
                "database": ["@../outside/secret.yaml"],
            },
        )

        config = Config(entry)

        assert config.get("database")["name"] == "stolen"

    def test_the_opt_out_admits_an_absolute_reference(self, tree: Path) -> None:
        entry = _write(
            tree / "configs" / "main.yaml",
            {
                "settings": {"allow_reference_outside_config_root": True},
                "database": [f"@{tree / 'outside' / 'secret.yaml'}"],
            },
        )

        config = Config(entry)

        assert config.get("database")["name"] == "stolen"

    def test_the_opt_out_is_not_applied_as_a_default_attribute(self, tree: Path) -> None:
        """Settings beside `config_root` are configuration, not config values.

        `apply_defaults` copies every dotless setting onto every atomic config
        as a default attribute, with `config_root` and friends excluded by
        name. A new sibling that is not excluded would silently appear as an
        `allow_reference_outside_config_root` key on every loaded object.
        """
        entry = _write(
            tree / "configs" / "main.yaml",
            {
                "settings": {"allow_reference_outside_config_root": True},
                "database": [{"name": "plain"}],
            },
        )

        config = Config(entry)

        assert "allow_reference_outside_config_root" not in config.get("database")

    def test_the_opt_out_logs_only_when_a_reference_actually_escapes(
        self, tree: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`find_config_file`'s convention: the warning marks a real escape.

        Warning on every load while the flag is on trains the reader to
        ignore it, which is the same as not warning at all.
        """
        entry = _write(
            tree / "configs" / "main.yaml",
            {
                "settings": {"allow_reference_outside_config_root": True},
                "database": ["@fragment.yaml"],
            },
        )

        with caplog.at_level("WARNING"):
            Config(entry)

        assert not [r for r in caplog.records if "outside" in r.getMessage()]

    def test_an_escape_under_the_opt_out_is_logged(
        self, tree: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        entry = _write(
            tree / "configs" / "main.yaml",
            {
                "settings": {"allow_reference_outside_config_root": True},
                "database": ["@../outside/secret.yaml"],
            },
        )

        with caplog.at_level("WARNING"):
            Config(entry)

        assert [r for r in caplog.records if "outside" in r.getMessage()]

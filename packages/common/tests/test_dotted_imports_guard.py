"""Tests for :func:`assert_no_ad_hoc_dotted_import`.

A source scan that matches nothing passes forever, so the cases that matter
most here are the ones asserting it *fails*. Mirrors the shape of the
error-text guard's own suite for the same reason: both are tools whose failure
mode is silence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_common.testing import assert_no_ad_hoc_dotted_import

COPY = """
import importlib


def resolve(ref):
    module_path, name = ref.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, name)
"""

CLEAN = """
from dataknobs_common.imports import resolve_callable


def resolve(ref):
    return resolve_callable(ref)
"""


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source)
    return path


def test_a_planted_copy_is_flagged(tmp_path: Path) -> None:
    """The mutation test. Without this the guard could scan nothing forever."""
    _write(tmp_path, "copy.py", COPY)

    with pytest.raises(AssertionError, match="import_module"):
        assert_no_ad_hoc_dotted_import(tmp_path)


def test_a_delegating_module_is_not_flagged(tmp_path: Path) -> None:
    _write(tmp_path, "clean.py", CLEAN)

    assert_no_ad_hoc_dotted_import(tmp_path)


def test_the_canonical_module_may_import_dynamically(tmp_path: Path) -> None:
    """It is the one place the operation is supposed to happen."""
    canonical = tmp_path / "dataknobs_common"
    canonical.mkdir()
    (canonical / "imports.py").write_text(COPY)

    assert_no_ad_hoc_dotted_import(tmp_path)


def test_dunder_import_is_flagged_too(tmp_path: Path) -> None:
    """Otherwise "stop failing the scan" and "write another copy" coincide."""
    _write(tmp_path, "sneaky.py", "def f(p):\n    return __import__(p)\n")

    with pytest.raises(AssertionError, match="__import__"):
        assert_no_ad_hoc_dotted_import(tmp_path)


def test_pkgutil_resolve_name_is_flagged_when_qualified(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "viapkgutil.py",
        "import pkgutil\n\n\ndef f(p):\n    return pkgutil.resolve_name(p)\n",
    )

    with pytest.raises(AssertionError, match="pkgutil.resolve_name"):
        assert_no_ad_hoc_dotted_import(tmp_path)


def test_an_unrelated_resolve_name_method_is_not_flagged(tmp_path: Path) -> None:
    """The false positive this guard shipped with, pinned so it cannot return.

    A config loader in this workspace has a ``resolve_name`` hook that maps a
    config *name* to a *path* — no import anywhere near it. Matching
    ``resolve_name`` on the bare attribute flagged that public API as a copy
    of this operation. The names collide; the operations do not.
    """
    _write(
        tmp_path,
        "loader.py",
        "class Loader:\n"
        "    def __init__(self, resolver):\n"
        "        self._resolver = resolver\n\n"
        "    def load(self, name):\n"
        "        return self._resolver.resolve_name(name)\n",
    )

    assert_no_ad_hoc_dotted_import(tmp_path)


def test_find_spec_is_not_flagged(tmp_path: Path) -> None:
    """It asks whether a module *could* be imported, without importing it.

    That is the probe an optional-dependency guard makes, and there are
    several in this workspace. Flagging them would push authors to add
    ``allow=`` entries for code that is not doing this operation at all,
    which is how an allow-list stops being readable.
    """
    _write(
        tmp_path,
        "probe.py",
        "import importlib.util\n\n\n"
        "def available(name):\n"
        "    return importlib.util.find_spec(name) is not None\n",
    )

    assert_no_ad_hoc_dotted_import(tmp_path)


def test_an_allowed_site_passes(tmp_path: Path) -> None:
    _write(tmp_path, "copy.py", COPY)

    assert_no_ad_hoc_dotted_import(tmp_path, allow={"copy.py:7"})


def test_an_allow_entry_matching_nothing_is_an_error(tmp_path: Path) -> None:
    """The hard-won rule, inherited from the error-text guard.

    A suppression whose site moved is a hole, and a silent one reads as a
    clean scan — the guard reports success precisely because it is no longer
    looking where the problem is.
    """
    _write(tmp_path, "clean.py", CLEAN)

    with pytest.raises(AssertionError, match="matched no flagged site"):
        assert_no_ad_hoc_dotted_import(tmp_path, allow={"clean.py:99"})


def test_an_allow_entry_matching_the_wrong_line_is_an_error(tmp_path: Path) -> None:
    """A copy that moved down two lines is not covered by its old entry."""
    _write(tmp_path, "copy.py", COPY)

    with pytest.raises(AssertionError, match="matched no flagged site"):
        assert_no_ad_hoc_dotted_import(tmp_path, allow={"copy.py:5"})


def test_every_finding_is_reported_not_just_the_first(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", COPY)
    _write(tmp_path, "b.py", COPY)

    with pytest.raises(AssertionError) as excinfo:
        assert_no_ad_hoc_dotted_import(tmp_path)

    message = str(excinfo.value)
    assert "a.py" in message
    assert "b.py" in message
    assert message.startswith("2 dynamic import(s)")


def test_an_unparseable_file_does_not_break_the_scan(tmp_path: Path) -> None:
    """A file that cannot be parsed cannot hold a copy either."""
    _write(tmp_path, "broken.py", "def (:\n")
    _write(tmp_path, "clean.py", CLEAN)

    assert_no_ad_hoc_dotted_import(tmp_path)

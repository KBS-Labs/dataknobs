"""Tests for the broad-except error-text source guard.

Half of these are mutation tests on the guard itself. A source scanner that
silently matches nothing reports success forever, which is worse than having no
guard at all, so each shape it is supposed to catch gets a positive case and
each shape it must *not* flag gets a negative one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_common.testing import assert_no_broad_except_in_error_text


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "sample.py"
    path.write_text(body, encoding="utf-8")
    return path


_ERRORS = frozenset({"ConfigurationError", "ConfigError"})


def test_flags_an_fstring_interpolating_a_broadly_caught_exception(tmp_path):
    """The shape the guard exists for."""
    path = _write(
        tmp_path,
        "def f():\n"
        "    try:\n"
        "        build()\n"
        "    except Exception as e:\n"
        '        raise ConfigurationError(f"failed: {e}") from e\n',
    )

    with pytest.raises(AssertionError) as excinfo:
        assert_no_broad_except_in_error_text(path, error_names=_ERRORS)

    assert "ConfigurationError" in str(excinfo.value)
    assert "sample.py:5" in str(excinfo.value)


def test_flags_str_of_the_exception_too(tmp_path):
    """``{str(e)}`` is the same disclosure with an extra call."""
    path = _write(
        tmp_path,
        "def f():\n"
        "    try:\n"
        "        build()\n"
        "    except Exception as e:\n"
        '        raise ConfigError(f"failed: {str(e)}") from e\n',
    )

    with pytest.raises(AssertionError):
        assert_no_broad_except_in_error_text(path, error_names=_ERRORS)


def test_flags_a_bare_except(tmp_path):
    """A bare ``except:`` is at least as broad as ``except Exception``."""
    path = _write(
        tmp_path,
        "def f():\n"
        "    try:\n"
        "        build()\n"
        "    except BaseException as e:\n"
        '        raise ConfigurationError(f"{e}")\n',
    )

    with pytest.raises(AssertionError):
        assert_no_broad_except_in_error_text(path, error_names=_ERRORS)


def test_allows_the_bounded_replacement(tmp_path):
    """The fixed shape: a class name, not the message.

    ``type(e).__name__`` reads the bound name but yields a class name, which
    is why the check is structural — a substring search for ``e`` would flag
    the fix as if it were the defect and make the guard unusable.
    """
    path = _write(
        tmp_path,
        "def f():\n"
        "    try:\n"
        "        build()\n"
        "    except Exception as e:\n"
        '        raise ConfigurationError(f"failed ({type(e).__name__})") from e\n',
    )

    assert_no_broad_except_in_error_text(path, error_names=_ERRORS)


def test_allows_a_narrow_except(tmp_path):
    """``ImportError`` text is module names — bounded, and worth keeping."""
    path = _write(
        tmp_path,
        "def f():\n"
        "    try:\n"
        "        build()\n"
        "    except ImportError as e:\n"
        '        raise ConfigurationError(f"failed: {e}") from e\n',
    )

    assert_no_broad_except_in_error_text(path, error_names=_ERRORS)


def test_allows_an_unlisted_error_type(tmp_path):
    """Only the named types are guarded; the rest are the caller's call."""
    path = _write(
        tmp_path,
        "def f():\n"
        "    try:\n"
        "        build()\n"
        "    except Exception as e:\n"
        '        raise RuntimeError(f"failed: {e}") from e\n',
    )

    assert_no_broad_except_in_error_text(path, error_names=_ERRORS)


def test_ignore_exempts_a_reviewed_site(tmp_path):
    """A site judged bounded can be exempted by file and line."""
    path = _write(
        tmp_path,
        "def f():\n"
        "    try:\n"
        "        build()\n"
        "    except Exception as e:\n"
        '        raise ConfigurationError(f"failed: {e}") from e\n',
    )

    assert_no_broad_except_in_error_text(
        path, error_names=_ERRORS, ignore={"sample.py:5"}
    )


def test_reports_every_site_not_just_the_first(tmp_path):
    """One run should show the whole surface."""
    path = _write(
        tmp_path,
        "def f():\n"
        "    try:\n"
        "        build()\n"
        "    except Exception as e:\n"
        '        raise ConfigurationError(f"a: {e}") from e\n'
        "def g():\n"
        "    try:\n"
        "        build()\n"
        "    except Exception as exc:\n"
        '        raise ConfigError(f"b: {exc}") from exc\n',
    )

    with pytest.raises(AssertionError) as excinfo:
        assert_no_broad_except_in_error_text(path, error_names=_ERRORS)

    message = str(excinfo.value)
    assert "2 error message(s)" in message
    assert "sample.py:5" in message
    assert "sample.py:10" in message


def test_scans_a_directory_recursively(tmp_path):
    """Roots may be directories; nested modules are covered."""
    nested = tmp_path / "pkg" / "sub"
    nested.mkdir(parents=True)
    (nested / "mod.py").write_text(
        "def f():\n"
        "    try:\n"
        "        build()\n"
        "    except Exception as e:\n"
        '        raise ConfigurationError(f"failed: {e}") from e\n',
        encoding="utf-8",
    )

    with pytest.raises(AssertionError) as excinfo:
        assert_no_broad_except_in_error_text(tmp_path, error_names=_ERRORS)

    assert "mod.py" in str(excinfo.value)


class TestCommonSourceIsClean:
    """The guard applied to this package."""

    def test_no_broad_except_in_configuration_error_text(self):
        src = Path(__file__).resolve().parents[1] / "src"

        assert_no_broad_except_in_error_text(
            src,
            error_names={"ConfigurationError", "PackResolutionError"},
        )

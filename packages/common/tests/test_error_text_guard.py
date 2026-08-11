"""Tests for the broad-except error-text source guard.

Half of these are mutation tests on the guard itself. A source scanner that
silently matches nothing reports success forever, which is worse than having no
guard at all, so each shape it is supposed to catch gets a positive case and
each shape it must *not* flag gets a negative one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_common.testing import (
    GUARDED_ERROR_NAMES,
    assert_no_broad_except_in_error_text,
)


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


def test_flags_base_exception_too(tmp_path):
    """``except BaseException`` is at least as broad as ``except Exception``.

    A bare ``except:`` is broader still, but binds no name, so there is no
    identifier for a message to interpolate and nothing for this guard to
    track. It is out of scope by construction, not by omission.
    """
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


def test_flags_import_error_by_default(tmp_path):
    """``except ImportError`` is narrow and still unbounded.

    Its text reads ``cannot import name 'X' from 'pkg'
    (/abs/path/site-packages/pkg/__init__.py)`` — an absolute filesystem
    path, which is exactly what a not-found error withholds on the grounds
    that it doubles as a map of the server's filesystem.

    Default rather than opt-in because the reason is a property of the
    exception type, not of the package catching it: an opt-in that every
    caller has to remember is a guard that narrows quietly, and this one was
    recommended in its own docstring and passed by nobody for as long as it
    existed.
    """
    path = _write(
        tmp_path,
        "def f():\n"
        "    try:\n"
        "        import_module(name)\n"
        "    except ImportError as e:\n"
        '        raise ConfigError(f"failed to import {name}: {e}") from e\n',
    )

    with pytest.raises(AssertionError) as excinfo:
        assert_no_broad_except_in_error_text(path, error_names=_ERRORS)

    assert "sample.py:5" in str(excinfo.value)


def test_a_narrow_except_is_still_not_flagged(tmp_path):
    """The default set stays a set, not "every named type".

    ``AttributeError`` from ``getattr`` yields a module and an attribute
    name — text the project can reason about. If widening the default had
    swept in every narrow clause, the guard would flag the prescribed fix
    along with the defect.

    This replaces a test that made the same point with ``ImportError`` and a
    docstring reading "bounded, and worth keeping" — which contradicted the
    module's own documentation, where ``ImportError`` was the named example
    of a narrow clause that is *not* bounded. The guard's suite and the
    guard's docs disagreed about that type from the day both were written.
    """
    path = _write(
        tmp_path,
        "def f():\n"
        "    try:\n"
        "        getattr(mod, name)\n"
        "    except AttributeError as e:\n"
        '        raise ConfigError(f"failed: {e}") from e\n',
    )

    assert_no_broad_except_in_error_text(path, error_names=_ERRORS)


def test_an_explicit_unbounded_types_still_overrides_the_default(tmp_path):
    """The parameter is an override, and stays one.

    Passing it replaces the default rather than extending it, so a caller
    that names only ``Exception`` gets exactly that — the behaviour every
    call site had before ``ImportError`` joined the default.
    """
    path = _write(
        tmp_path,
        "def f():\n"
        "    try:\n"
        "        import_module(name)\n"
        "    except ImportError as e:\n"
        '        raise ConfigError(f"failed: {e}") from e\n',
    )

    assert_no_broad_except_in_error_text(path, error_names=_ERRORS, unbounded_types={"Exception"})


def test_flags_a_qualified_broad_except(tmp_path):
    """``except builtins.Exception`` is the same clause spelled longer."""
    path = _write(
        tmp_path,
        "import builtins\n"
        "def f():\n"
        "    try:\n"
        "        build()\n"
        "    except builtins.Exception as e:\n"
        '        raise ConfigurationError(f"failed: {e}") from e\n',
    )

    with pytest.raises(AssertionError):
        assert_no_broad_except_in_error_text(path, error_names=_ERRORS)


class TestTheShapesThatAreNotAnFString:
    """Every way to get the exception's text into the message.

    The guard's first cut recognised one syntactic form — an f-string sitting
    directly in a positional argument — and reported green on the other five.
    Three of the sites it was written to protect used the first case below, so
    it passed against the very code that motivated it. Interpolation is not a
    syntax; it is any read of the caught name that is not provably a class
    name, and each of these is a way to spell it.
    """

    def test_an_intermediate_variable(self, tmp_path):
        """``msg = f"...{e}"`` then ``raise X(msg)`` — the historical shape."""
        path = _write(
            tmp_path,
            "def f():\n"
            "    try:\n"
            "        build()\n"
            "    except Exception as e:\n"
            '        msg = f"failed: {e}"\n'
            "        raise ConfigurationError(msg) from e\n",
        )

        with pytest.raises(AssertionError) as excinfo:
            assert_no_broad_except_in_error_text(path, error_names=_ERRORS)

        assert "sample.py:6" in str(excinfo.value)

    def test_a_rebound_alias(self, tmp_path):
        """Assigning the exception to another name does not launder it."""
        path = _write(
            tmp_path,
            "def f():\n"
            "    try:\n"
            "        build()\n"
            "    except Exception as e:\n"
            "        err = e\n"
            '        raise ConfigurationError(f"failed: {err}") from e\n',
        )

        with pytest.raises(AssertionError):
            assert_no_broad_except_in_error_text(path, error_names=_ERRORS)

    def test_a_keyword_argument(self, tmp_path):
        """``context=`` is disclosed for several types, so it is in scope.

        ``ValidationError`` is ``ErrorPolicy(422, True, True)`` — its context
        is returned to the caller — so a message-only scan guards the safer
        half of that row and leaves the other open.
        """
        path = _write(
            tmp_path,
            "def f():\n"
            "    try:\n"
            "        build()\n"
            "    except Exception as e:\n"
            "        raise ConfigurationError(\n"
            '            "bounded", context={"detail": f"{e}"}\n'
            "        ) from e\n",
        )

        with pytest.raises(AssertionError):
            assert_no_broad_except_in_error_text(path, error_names=_ERRORS)

    def test_a_bare_str_call(self, tmp_path):
        """``raise X(str(e))`` has no f-string at all."""
        path = _write(
            tmp_path,
            "def f():\n"
            "    try:\n"
            "        build()\n"
            "    except Exception as e:\n"
            "        raise ConfigurationError(str(e)) from e\n",
        )

        with pytest.raises(AssertionError):
            assert_no_broad_except_in_error_text(path, error_names=_ERRORS)

    def test_percent_formatting(self, tmp_path):
        path = _write(
            tmp_path,
            "def f():\n"
            "    try:\n"
            "        build()\n"
            "    except Exception as e:\n"
            '        raise ConfigurationError("failed: %s" % e) from e\n',
        )

        with pytest.raises(AssertionError):
            assert_no_broad_except_in_error_text(path, error_names=_ERRORS)

    def test_str_format(self, tmp_path):
        path = _write(
            tmp_path,
            "def f():\n"
            "    try:\n"
            "        build()\n"
            "    except Exception as e:\n"
            '        raise ConfigurationError("failed: {}".format(e)) from e\n',
        )

        with pytest.raises(AssertionError):
            assert_no_broad_except_in_error_text(path, error_names=_ERRORS)

    def test_concatenation(self, tmp_path):
        path = _write(
            tmp_path,
            "def f():\n"
            "    try:\n"
            "        build()\n"
            "    except Exception as e:\n"
            '        raise ConfigurationError("failed: " + str(e)) from e\n',
        )

        with pytest.raises(AssertionError):
            assert_no_broad_except_in_error_text(path, error_names=_ERRORS)

    def test_reading_args(self, tmp_path):
        """``exc.args[0]`` is the message by another route."""
        path = _write(
            tmp_path,
            "def f():\n"
            "    try:\n"
            "        build()\n"
            "    except Exception as e:\n"
            '        raise ConfigurationError(f"failed: {e.args[0]}") from e\n',
        )

        with pytest.raises(AssertionError):
            assert_no_broad_except_in_error_text(path, error_names=_ERRORS)

    def test_repr(self, tmp_path):
        """``repr`` includes the message and adds the type."""
        path = _write(
            tmp_path,
            "def f():\n"
            "    try:\n"
            "        build()\n"
            "    except Exception as e:\n"
            '        raise ConfigurationError(f"failed: {e!r}") from e\n',
        )

        with pytest.raises(AssertionError):
            assert_no_broad_except_in_error_text(path, error_names=_ERRORS)

    def test_a_helper_one_frame_away(self, tmp_path):
        """Passing the exception to a function is not evidence of safety.

        The guard cannot see what ``_describe`` does, so it assumes the worst.
        A helper that is genuinely bounded is what ``ignore=`` is for.
        """
        path = _write(
            tmp_path,
            "def f():\n"
            "    try:\n"
            "        build()\n"
            "    except Exception as e:\n"
            "        raise ConfigurationError(_describe(e)) from e\n",
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


class TestTheFixedShapesStayUnflagged:
    """Inverting to fail-closed must not make the prescribed fix look wrong.

    The guard now flags any read of the caught name it cannot prove safe, so
    the forms that *are* safe need pinning from the other side — otherwise the
    first person to apply the documented fix gets a failure for doing it.
    """

    def test_the_class_name(self, tmp_path):
        path = _write(
            tmp_path,
            "def f():\n"
            "    try:\n"
            "        build()\n"
            "    except Exception as e:\n"
            "        raise ConfigurationError(\n"
            '            f"failed ({type(e).__name__})"\n'
            "        ) from e\n",
        )

        assert_no_broad_except_in_error_text(path, error_names=_ERRORS)

    def test_the_class_name_by_dunder_class(self, tmp_path):
        path = _write(
            tmp_path,
            "def f():\n"
            "    try:\n"
            "        build()\n"
            "    except Exception as e:\n"
            "        raise ConfigurationError(\n"
            '            f"failed ({e.__class__.__name__})"\n'
            "        ) from e\n",
        )

        assert_no_broad_except_in_error_text(path, error_names=_ERRORS)

    def test_the_class_name_through_a_local(self, tmp_path):
        """The taint tracking must not treat a class name as tainted."""
        path = _write(
            tmp_path,
            "def f():\n"
            "    try:\n"
            "        build()\n"
            "    except Exception as e:\n"
            "        kind = type(e).__name__\n"
            '        raise ConfigurationError(f"failed ({kind})") from e\n',
        )

        assert_no_broad_except_in_error_text(path, error_names=_ERRORS)

    def test_an_isinstance_branch(self, tmp_path):
        """Classifying the exception yields a bool, not its text."""
        path = _write(
            tmp_path,
            "def f():\n"
            "    try:\n"
            "        build()\n"
            "    except Exception as e:\n"
            "        if isinstance(e, KeyError):\n"
            '            raise ConfigurationError("missing key") from e\n',
        )

        assert_no_broad_except_in_error_text(path, error_names=_ERRORS)

    def test_raise_from_is_the_prescribed_fix(self, tmp_path):
        """``from exc`` is where the original is supposed to go."""
        path = _write(
            tmp_path,
            "def f():\n"
            "    try:\n"
            "        build()\n"
            "    except Exception as e:\n"
            '        raise ConfigurationError("bounded") from e\n',
        )

        assert_no_broad_except_in_error_text(path, error_names=_ERRORS)

    def test_logging_the_full_text_is_fine(self, tmp_path):
        """The log is exactly where the unbounded text is wanted."""
        path = _write(
            tmp_path,
            "def f():\n"
            "    try:\n"
            "        build()\n"
            "    except Exception as e:\n"
            '        logger.warning("failed: %s", e)\n'
            '        raise ConfigurationError("bounded") from e\n',
        )

        assert_no_broad_except_in_error_text(path, error_names=_ERRORS)


class TestIgnoreEntries:
    """``ignore=`` is a suppression, so a stale one must not stay silent."""

    def test_exempts_a_reviewed_site(self, tmp_path):
        path = _write(
            tmp_path,
            "def f():\n"
            "    try:\n"
            "        build()\n"
            "    except Exception as e:\n"
            '        raise ConfigurationError(f"failed: {e}") from e\n',
        )

        assert_no_broad_except_in_error_text(path, error_names=_ERRORS, ignore={"sample.py:5"})

    def test_a_path_qualified_entry_matches(self, tmp_path):
        """The docstring promises a path suffix; a basename-only match broke it.

        ``pkg/mod.py:5`` silently matched nothing, so a suppression its author
        believed active was not — and the bare basename it forced instead
        exempts that line in *every* file of that name under the root.
        """
        nested = tmp_path / "pkg"
        nested.mkdir()
        (nested / "mod.py").write_text(
            "def f():\n"
            "    try:\n"
            "        build()\n"
            "    except Exception as e:\n"
            '        raise ConfigurationError(f"failed: {e}") from e\n',
            encoding="utf-8",
        )

        assert_no_broad_except_in_error_text(tmp_path, error_names=_ERRORS, ignore={"pkg/mod.py:5"})

    def test_an_entry_that_matches_nothing_fails(self, tmp_path):
        """A suppression for a site that moved is a hole, reported as one."""
        path = _write(
            tmp_path,
            "def f():\n"
            "    try:\n"
            "        build()\n"
            "    except Exception as e:\n"
            '        raise ConfigurationError("bounded") from e\n',
        )

        with pytest.raises(AssertionError) as excinfo:
            assert_no_broad_except_in_error_text(path, error_names=_ERRORS, ignore={"sample.py:99"})

        assert "sample.py:99" in str(excinfo.value)


def test_an_unparseable_file_is_named(tmp_path):
    """One bad file should not abort the scan with a raw SyntaxError."""
    (tmp_path / "broken.py").write_text("def f(:\n", encoding="utf-8")

    with pytest.raises(AssertionError) as excinfo:
        assert_no_broad_except_in_error_text(tmp_path, error_names=_ERRORS)

    assert "broken.py" in str(excinfo.value)


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
            error_names=GUARDED_ERROR_NAMES | {"PackResolutionError"},
        )

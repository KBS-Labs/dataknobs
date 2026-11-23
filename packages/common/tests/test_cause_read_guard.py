"""``assert_no_cause_read`` and the error-text guard's view through a mark."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_common.testing import (
    assert_no_broad_except_in_error_text,
    assert_no_cause_read,
)


def _write(tmp_path: Path, source: str, name: str = "helper.py") -> Path:
    path = tmp_path / "pkg" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


class TestAssertNoCauseRead:
    def test_a_handler_reading_the_cause_is_flagged(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            "def create():\n"
            "    try:\n"
            "        return build()\n"
            "    except OperationError as exc:\n"
            "        raise exc.__cause__\n",
        )
        with pytest.raises(AssertionError, match=r"helper\.py:5 in create"):
            assert_no_cause_read(tmp_path)

    def test_a_helper_taking_the_exception_as_a_parameter_is_flagged(self, tmp_path: Path) -> None:
        """Not only handlers: the read that motivated a whole-load scan was one of these."""
        _write(tmp_path, "def fault(exc):\n    return str(exc.__cause__)\n")
        with pytest.raises(AssertionError, match=r"in fault"):
            assert_no_cause_read(tmp_path)

    def test_reading_only_the_causes_type_is_not_flagged(self, tmp_path: Path) -> None:
        _write(tmp_path, "def name(exc):\n    return type(exc.__cause__).__name__\n")
        assert_no_cause_read(tmp_path)

    def test_an_allowed_function_is_not_flagged(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            "class Handler:\n    def log(self, exc):\n        return exc.__cause__\n",
        )
        assert_no_cause_read(tmp_path, allowed={"pkg/helper.py::Handler.log"})

    def test_an_allowed_file_is_not_flagged(self, tmp_path: Path) -> None:
        _write(tmp_path, "def a(exc):\n    return exc.__cause__\n")
        assert_no_cause_read(tmp_path, allowed={"pkg/helper.py"})

    def test_an_allowed_entry_matching_nothing_fails(self, tmp_path: Path) -> None:
        _write(tmp_path, "def a(exc):\n    return exc\n")
        with pytest.raises(AssertionError, match="matched no `__cause__` read"):
            assert_no_cause_read(tmp_path, allowed={"pkg/helper.py::gone"})

    def test_reading_the_context_is_flagged(self, tmp_path: Path) -> None:
        """A registry raises inside the handler, so ``__context__`` holds the same error."""
        _write(tmp_path, "def fault(exc):\n    return str(exc.__context__)\n")
        with pytest.raises(AssertionError, match=r"in fault \(__context__\)"):
            assert_no_cause_read(tmp_path)

    def test_naming_the_cause_as_a_string_is_flagged(self, tmp_path: Path) -> None:
        _write(tmp_path, "def fault(exc):\n    return getattr(exc, '__cause__')\n")
        with pytest.raises(AssertionError, match=r"in fault \('__cause__'\)"):
            assert_no_cause_read(tmp_path)

    @pytest.mark.parametrize(
        "source",
        [
            "import traceback\ndef fault(exc):\n    return traceback.format_exception(exc)\n",
            "from traceback import format_exc\n",
        ],
    )
    def test_formatting_the_chain_is_flagged(self, tmp_path: Path, source: str) -> None:
        _write(tmp_path, source)
        with pytest.raises(AssertionError, match=r"traceback\.format_exc"):
            assert_no_cause_read(tmp_path)

    def test_assigning_the_cause_is_not_a_read(self, tmp_path: Path) -> None:
        _write(tmp_path, "def a(exc, other):\n    exc.__cause__ = other\n")
        assert_no_cause_read(tmp_path)


class TestTheErrorTextGuardSeesThroughAMark:
    """Wrapping a raise in a marking call must not hide it from the guard."""

    def test_a_marked_raise_interpolating_a_broad_exception_is_still_flagged(
        self, tmp_path: Path
    ) -> None:
        _write(
            tmp_path,
            "def build():\n"
            "    try:\n"
            "        return make()\n"
            "    except Exception as exc:\n"
            "        raise _written_by_dataknobs(ConfigurationError(f'failed: {exc}'))\n",
        )
        with pytest.raises(AssertionError, match="ConfigurationError"):
            assert_no_broad_except_in_error_text(tmp_path, error_names={"ConfigurationError"})

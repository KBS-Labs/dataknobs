"""Tests for the condition behind the ``requires_real_*`` markers.

The markers themselves are module-level constants, evaluated once when
``_core`` is imported, so a test cannot give one an environment of its own.
:func:`must_skip_real_service` is the part that can be driven, and it holds
the whole contract: three terms, any one of which is enough to skip.

What these pin is the *conjunction*. The shape they replace across the
suites checked only the opt-in variable, which is the term that never
catches anything interesting -- a suite that opts in against a **down**
server got connection errors where a skip naming the cause is the honest
answer, and a suite whose optional driver was absent got an ImportError
reported as a failure.
"""

from __future__ import annotations

import pytest

from dataknobs_common.testing import (
    must_skip_real_service,
    requires_real_elasticsearch,
    requires_real_postgres,
    requires_real_postgres_sync,
    requires_real_s3,
)

# `pytest` is installed wherever this suite runs, so it stands in for a
# driver that is definitely importable. `dataknobs_no_such_driver` stands
# in for one that is definitely not.
INSTALLED = "pytest"
ABSENT = "dataknobs_no_such_driver"


@pytest.fixture(autouse=True)
def _clear_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the opt-in variables the ambient environment may carry."""
    for var in ("TEST_POSTGRES", "TEST_ELASTICSEARCH", "TEST_S3"):
        monkeypatch.delenv(var, raising=False)


class TestEachTermCanSkipAlone:
    """Any one of the three terms is sufficient to skip."""

    def test_all_three_satisfied_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_THING", "true")
        assert (
            must_skip_real_service(opt_in_var="TEST_THING", reachable=True, package=INSTALLED)
            is False
        )

    def test_unreachable_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_THING", "true")
        assert (
            must_skip_real_service(opt_in_var="TEST_THING", reachable=False, package=INSTALLED)
            is True
        )

    def test_not_opted_in_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TEST_THING", raising=False)
        assert (
            must_skip_real_service(opt_in_var="TEST_THING", reachable=True, package=INSTALLED)
            is True
        )

    def test_absent_driver_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_THING", "true")
        assert (
            must_skip_real_service(opt_in_var="TEST_THING", reachable=True, package=ABSENT) is True
        )


class TestOptInSpelling:
    """The opt-in compares case-insensitively against exactly ``true``."""

    @pytest.mark.parametrize("value", ["true", "TRUE", "True", "tRuE"])
    def test_true_in_any_case_opts_in(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("TEST_THING", value)
        assert (
            must_skip_real_service(opt_in_var="TEST_THING", reachable=True, package=INSTALLED)
            is False
        )

    @pytest.mark.parametrize("value", ["", "1", "yes", "on", "false", "  true  "])
    def test_anything_else_does_not(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        """Only the literal word opts in.

        ``"1"`` and ``"yes"`` are deliberately *not* accepted: the suites
        this replaces compared against ``"true"`` alone, so widening here
        would opt in runs that are set up to stay out.
        """
        monkeypatch.setenv("TEST_THING", value)
        assert (
            must_skip_real_service(opt_in_var="TEST_THING", reachable=True, package=INSTALLED)
            is True
        )


class TestTheMarkersAreBuiltFromIt:
    """Each published marker is a skipif carrying a reason that names all three."""

    @pytest.mark.parametrize(
        ("marker", "service_word", "driver"),
        [
            (requires_real_postgres, "TEST_POSTGRES=true", "asyncpg"),
            (requires_real_postgres_sync, "TEST_POSTGRES=true", "psycopg2"),
            (requires_real_elasticsearch, "TEST_ELASTICSEARCH=true", "elasticsearch"),
            (requires_real_s3, "TEST_S3=true", "boto3"),
        ],
    )
    def test_reason_names_opt_in_and_driver(
        self, marker: pytest.MarkDecorator, service_word: str, driver: str
    ) -> None:
        reason = marker.kwargs["reason"]
        assert service_word in reason
        assert driver in reason
        assert "reachable" in reason

    def test_the_two_postgres_markers_differ_only_in_driver(self) -> None:
        """The distinction the pair exists for.

        A sync suite reaches Postgres through psycopg2 and an async one
        through asyncpg. Both drivers arrive together in
        ``dataknobs-data[postgres]``, so gating one suite on the other's
        driver would never actually misfire in this repo -- and would
        still be stating a requirement the suite does not have.
        """
        assert "asyncpg" in requires_real_postgres.kwargs["reason"]
        assert "psycopg2" not in requires_real_postgres.kwargs["reason"]
        assert "psycopg2" in requires_real_postgres_sync.kwargs["reason"]
        assert "asyncpg" not in requires_real_postgres_sync.kwargs["reason"]

"""The provisioning record counts tests that took a fresh index and never wrote to it.

A test that only reads does not need an index of its own: it pays for one in
setup (create, seed, refresh) and could have shared a seeded corpus. The record
the plugin writes when ``DK_PROVISIONING_RECORD`` is set counts those tests per
file, and the quality contract holds that count under a ceiling.

The property spans a whole session, from a fixture's setup to the record
written at its end, so these tests run an inner pytest session against the
real dev node. Determinism is checked too: a ratchet over a count that moved
with the number of xdist workers would fail at random.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dataknobs_common.testing import requires_elasticsearch

pytest_plugins = ("pytester",)

pytestmark = requires_elasticsearch

_INNER_INI = """
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
"""

#: One file, one test of each shape the count must tell apart.
_INNER_TESTS = """
import json

import pytest
from dataknobs_utils.requests_utils import HEADERS, RequestHelper


def _helper(config):
    return RequestHelper(config["host"], config["port"], timeout=5, headers=HEADERS)


def _write(config, doc_id):
    response = _helper(config).put(
        f"{config['index']}/_doc/{doc_id}", payload=json.dumps({"n": 1}),
        params={"refresh": "true"},
    )
    assert response.succeeded, response.json


@pytest.fixture
def seeded(make_elasticsearch_test_index):
    for config in make_elasticsearch_test_index("test_count_"):
        _write(config, "a")
        yield config


@pytest.fixture
def named_only(make_elasticsearch_test_index):
    yield from make_elasticsearch_test_index("test_count_")


def test_reads_only(seeded):
    assert _helper(seeded).get(f"{seeded['index']}/_count").json["count"] == 1


def test_writes(seeded):
    _write(seeded, "b")


def test_takes_a_name_it_never_creates(named_only):
    assert named_only["index"].startswith("test_count_")


def test_takes_nothing():
    pass
"""


def _run(pytester: pytest.Pytester, record: Path, *args: str) -> dict[str, Any]:
    pytester.makeini(_INNER_INI)
    pytester.makepyfile(test_inner=_INNER_TESTS)
    result = pytester.runpytest_subprocess("-p", "no:randomly", "-p", "no:cacheprovider", *args)
    result.assert_outcomes(passed=4)
    return json.loads(record.read_text())


@pytest.fixture
def record(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = pytester.path / "provisioning.json"
    monkeypatch.setenv("DK_PROVISIONING_RECORD", str(path))
    # Under the gate this test runs on an xdist worker, whose variable the
    # inner session would inherit and read as "I am a worker; do not record".
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    return path


def test_a_read_only_test_on_a_seeded_fresh_index_is_counted(
    pytester: pytest.Pytester, record: Path
) -> None:
    written = _run(pytester, record)
    assert written["files"] == {
        "test_inner.py": {"took": 3, "unwritten": ["test_inner.py::test_reads_only"]}
    }
    assert written["service_skipped"] == {"elasticsearch": 0}


def test_the_count_does_not_depend_on_the_number_of_workers(
    pytester: pytest.Pytester, record: Path
) -> None:
    serial = _run(pytester, record)
    record.unlink()
    parallel = _run(pytester, record, "-n", "2")
    assert parallel == serial


def test_nothing_is_written_without_the_variable(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DK_PROVISIONING_RECORD", raising=False)
    pytester.makeini(_INNER_INI)
    pytester.makepyfile(test_inner=_INNER_TESTS)
    pytester.runpytest_subprocess("-p", "no:randomly").assert_outcomes(passed=4)
    assert not list(pytester.path.glob("*.json"))


def test_a_test_skipped_for_an_unreachable_node_is_recorded(
    pytester: pytest.Pytester, record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A count taken while the node was down would fall, and ask for a ceiling unearned.

    The record says how many tests the node's absence skipped, so the contract
    can refuse to read it as a count.
    """
    monkeypatch.setenv("ELASTICSEARCH_PORT", "1")
    pytester.makeini(_INNER_INI)
    pytester.makepyfile(
        test_inner="""
        from dataknobs_common.testing import requires_elasticsearch, requires_real_elasticsearch

        @requires_elasticsearch
        def test_one():
            pass

        @requires_real_elasticsearch
        def test_two():
            pass

        def test_three():
            pass
        """
    )
    result = pytester.runpytest_subprocess("-p", "no:randomly", "-rs")
    result.assert_outcomes(passed=1, skipped=2)
    assert json.loads(record.read_text())["service_skipped"] == {"elasticsearch": 2}

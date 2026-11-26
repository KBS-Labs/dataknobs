"""A seeded corpus is built once and shared by the tests that only read it.

``make_elasticsearch_corpus`` seeds an index the first time a test module
asks for a corpus by name, and hands every later test in that module the
same index. Sharing is only safe while nobody writes, so after each test the
plugin compares the corpus with how it was left after seeding. A test that
wrote errors in its own teardown, and every later user errors at setup,
naming that test, rather than failing on data it did not expect.

These properties span a session (a seed paid once, a check after a teardown,
an index deleted at the end), so each test runs an inner pytest session
against the real dev node, in a subprocess as ``test_provisioning_count.py``
does.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from dataknobs_common.testing import elasticsearch_index_fingerprint, requires_elasticsearch

pytest_plugins = ("pytester",)

pytestmark = requires_elasticsearch

_INNER_INI = """
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
"""

#: Helpers every inner module shares: a seed that leaves a trace of each call
#: in ``seeds.txt``, so the outer test can count them, and one write.
_HELPERS = """
import json
from pathlib import Path

from dataknobs_utils.requests_utils import HEADERS, RequestHelper


def helper(config):
    return RequestHelper(config["host"], config["port"], timeout=5, headers=HEADERS)


def write(config, doc_id):
    response = helper(config).put(
        f"{config['index']}/_doc/{doc_id}", payload=json.dumps({"n": 1}),
        params={"refresh": "true"},
    )
    assert response.succeeded, response.json


def count(config):
    return helper(config).get(f"{config['index']}/_count").json["count"]


def seed(config):
    with Path("seeds.txt").open("a") as trace:
        trace.write(config["index"] + "\\n")
    write(config, "seeded")
"""


def _run(pytester: pytest.Pytester, *args: str, **modules: str) -> pytest.RunResult:
    pytester.makeini(_INNER_INI)
    pytester.makepyfile(helpers=_HELPERS, **modules)
    return pytester.runpytest_subprocess("-p", "no:randomly", "-p", "no:cacheprovider", *args)


def _seeded(pytester: pytest.Pytester) -> list[str]:
    trace = pytester.path / "seeds.txt"
    return trace.read_text().split() if trace.exists() else []


def _exists(params: dict[str, Any], index: str) -> bool:
    return elasticsearch_index_fingerprint(params["host"], params["port"], index).exists


_READERS = """
import pytest
from helpers import count, seed


@pytest.fixture
def corpus(make_elasticsearch_corpus):
    return make_elasticsearch_corpus("test_corpus_", seed)


def test_one(corpus):
    assert count(corpus) == 1


def test_two(corpus):
    assert count(corpus) == 1


def test_three(corpus):
    assert count(corpus) == 1
"""


def test_readers_share_one_seeded_index_that_is_gone_after_the_session(
    pytester: pytest.Pytester, elasticsearch_connection_params: dict[str, Any]
) -> None:
    _run(pytester, test_readers=_READERS).assert_outcomes(passed=3)

    [index] = _seeded(pytester)
    assert index.startswith("test_corpus_")
    assert not _exists(elasticsearch_connection_params, index)


def test_each_module_has_its_own_corpus_of_a_name(pytester: pytest.Pytester) -> None:
    """Two modules that pick one name must not read each other's records."""
    _run(pytester, test_a=_READERS, test_b=_READERS).assert_outcomes(passed=6)

    assert len(set(_seeded(pytester))) == 2


def test_the_count_is_the_same_on_xdist_workers(pytester: pytest.Pytester) -> None:
    """Each worker seeds its own copy, and its readers still share it."""
    _run(pytester, "-n", "2", test_a=_READERS, test_b=_READERS).assert_outcomes(passed=6)

    assert 2 <= len(set(_seeded(pytester))) <= 4


_WRITER_THEN_READER = """
import pytest
from helpers import count, seed, write


@pytest.fixture
def corpus(make_elasticsearch_corpus):
    return make_elasticsearch_corpus("test_corpus_", seed)


def test_a_reads(corpus):
    assert count(corpus) == 1


def test_b_writes(corpus):
    write(corpus, "extra")


def test_c_reads(corpus):
    assert count(corpus) == 1
"""


def test_a_writer_errors_and_so_does_every_later_user_naming_it(
    pytester: pytest.Pytester,
) -> None:
    result = _run(pytester, "-rE", test_corpus=_WRITER_THEN_READER)

    # The writer's call passes and its teardown errors; the reader after it
    # errors at setup. The reader before it is untouched.
    result.assert_outcomes(passed=2, errors=2)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_b_writes*",
            "*test_corpus.py::test_b_writes wrote to the shared index test_corpus_*"
            "make_elasticsearch_test_index*",
            "*ERROR at setup of test_c_reads*",
            "*test_corpus_* was written by test_corpus.py::test_b_writes*",
        ]
    )


_FIXTURE_WRITES_ON_ITS_WAY_OUT = """
import pytest
from helpers import count, seed, write


@pytest.fixture
def corpus(make_elasticsearch_corpus):
    config = make_elasticsearch_corpus("test_corpus_", seed)
    yield config
    write(config, "cleanup")


def test_reads(corpus):
    assert count(corpus) == 1
"""


def test_a_write_in_a_fixture_s_teardown_is_caught(pytester: pytest.Pytester) -> None:
    result = _run(pytester, "-rE", test_corpus=_FIXTURE_WRITES_ON_ITS_WAY_OUT)

    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*test_corpus.py::test_reads wrote to the shared index*"])


_SEED_FAILS = """
import pytest
from helpers import write


def seed(config):
    write(config, "partial")
    raise RuntimeError("the seed failed")


def test_uses(make_elasticsearch_corpus):
    make_elasticsearch_corpus("test_corpus_bad_", seed)
"""


def test_a_failed_seed_leaves_no_index_behind(
    pytester: pytest.Pytester, elasticsearch_connection_params: dict[str, Any]
) -> None:
    result = _run(pytester, test_corpus=_SEED_FAILS)

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*the seed failed*"])
    # The test reports the index it was seeding; it must be deleted already.
    names = [
        word.strip("'\",.")
        for word in result.stdout.str().split()
        if word.strip("'\",.").startswith("test_corpus_bad_")
    ]
    assert names, result.stdout.str()
    assert not any(_exists(elasticsearch_connection_params, name) for name in names)


_CORPUS_AND_FRESH = """
import pytest
from helpers import count, seed, write


@pytest.fixture
def corpus(make_elasticsearch_corpus):
    return make_elasticsearch_corpus("test_corpus_", seed)


@pytest.fixture
def fresh(make_elasticsearch_test_index):
    for config in make_elasticsearch_test_index("test_fresh_"):
        write(config, "a")
        yield config


def test_reads_the_corpus(corpus):
    assert count(corpus) == 1


def test_reads_a_fresh_index(fresh):
    assert count(fresh) == 1
"""


def test_a_corpus_user_is_not_counted(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The count is of fresh indices nobody wrote to; a corpus is meant to be read."""
    record = pytester.path / "provisioning.json"
    monkeypatch.setenv("DK_PROVISIONING_RECORD", str(record))
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)

    _run(pytester, test_inner=_CORPUS_AND_FRESH).assert_outcomes(passed=2)

    assert json.loads(record.read_text())["files"] == {
        "test_inner.py": {
            "took": 1,
            "unwritten": ["test_inner.py::test_reads_a_fresh_index"],
        }
    }


def test_the_name_must_be_in_the_namespace_the_sweep_reclaims(
    pytester: pytest.Pytester,
) -> None:
    """A corpus a killed run leaks is reclaimed only if its name starts ``test_``."""
    result = _run(
        pytester,
        test_corpus="""
from helpers import seed


def test_uses(make_elasticsearch_corpus):
    make_elasticsearch_corpus("corpus_", seed)
""",
    )
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*must start 'test_'*got 'corpus_'*"])
    assert _seeded(pytester) == []

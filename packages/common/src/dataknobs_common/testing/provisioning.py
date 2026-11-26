# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Count tests that provisioned a fresh service resource and never wrote to it.

A test that only reads does not need a resource of its own. A fresh
Elasticsearch index costs a create, a seed and a refresh in setup, and a test
that then only searches it could have shared one seeded index with its
neighbours. This module counts such tests, per file, so the count can be held
under a ceiling. A count, unlike wall time, does not move with how loaded the
machine is.

This module is a pytest11 plugin (registered in
``packages/common/pyproject.toml``). It does nothing unless the environment
variable ``DK_PROVISIONING_RECORD`` names a file; the quality gate sets it for
each suite it runs. When set:

* A factory fixture that hands out a fresh resource calls
  :func:`take_fresh_resource` with a fingerprint function.
  ``make_elasticsearch_test_index`` does.
* As each test's call phase begins and ends, every resource it took is
  fingerprinted. A resource that existed when the call began and reads the
  same when it ended was provisioned in setup and never written to, and the
  test is counted. A resource that did not exist yet was never provisioned,
  so a test that only took a name costs nothing and is not counted.
* At the end of the session the controller writes the record::

      {"version": 1,
       "files": {"packages/data/tests/x.py": {"took": 4, "unwritten": ["<nodeid>"]}},
       "service_skipped": {"elasticsearch": 0}}

  ``files`` holds every file whose tests took a fresh resource, including
  those whose count is zero, so "measured zero" and "measured nothing" are
  different records. Paths are relative to the git root when there is one,
  and to pytest's rootdir otherwise. ``service_skipped`` counts tests that a
  service marker skipped because the service was unreachable: a count taken
  then would fall, and must not be read as one.

The result travels on each report's ``user_properties``, which cross the
pytest-xdist boundary, so the record is the same at ``-n 0`` and ``-n 8``.

The plugin also guards the other side of the trade: a resource seeded once
and **shared** by tests that only read it. A factory hands one out with
:func:`take_shared_resource` and, in its own teardown, calls
:func:`check_shared_resources`; ``make_elasticsearch_corpus`` does. Every
shared resource the test took is fingerprinted against the reading taken when
it was first handed out. A test that wrote errors in its own teardown, and
every later test that asks for the resource errors at setup, naming the
writer, rather than failing on data it did not expect. That check always
runs: it is what makes sharing safe, not a measurement.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The environment variable naming the record file. Unset, the plugin is inert.
RECORD_ENV = "DK_PROVISIONING_RECORD"

#: The record's format version.
RECORD_VERSION = 1

_COUNT_PROPERTY = "dk_provisioning"
_SKIP_PROPERTY = "dk_service_skip"


@dataclass(frozen=True)
class _Handle:
    """One fresh resource a test took: a name to report, and how to read it."""

    label: str
    fingerprint: Callable[[], Hashable | None]


@dataclass
class _Shared:
    """One shared resource: how to read it, how it read when first handed out, and who wrote it."""

    fingerprint: Callable[[], Hashable | None]
    baseline: Hashable | None
    instead: str
    written_by: str | None = None


class SharedResourceWrittenError(Exception):
    """A test wrote to a resource it shared with other tests."""


@dataclass
class _ProcessState:
    """What the plugin knows in this process. Per process, which is what xdist workers are.

    ``current_item`` is the test running now, set for the whole of its protocol
    (setup, call, teardown), so a factory called from any fixture, or through
    ``request.getfixturevalue``, is attributed to the test that asked.

    ``taken`` holds the handles each running test took, keyed by node id, and
    is cleared as each test ends.

    ``shared`` holds every shared resource handed out in this process, by label,
    and ``shared_taken`` the labels each running test took.

    ``recorder_active`` says a recorder is registered in this process. A pytest
    session run in-process inside a test (``pytester.runpytest_inprocess``) sees
    the same environment variable, and must not write the outer session's record.
    """

    current_item: Any = None
    taken: dict[str, list[_Handle]] = field(default_factory=dict)
    shared: dict[str, _Shared] = field(default_factory=dict)
    shared_taken: dict[str, list[str]] = field(default_factory=dict)
    recorder_active: bool = False


_state = _ProcessState()


def recording() -> bool:
    """Whether this session writes a provisioning record."""
    return bool(os.environ.get(RECORD_ENV))


def take_fresh_resource(label: str, fingerprint: Callable[[], Hashable | None]) -> None:
    """Record that the running test took a fresh resource.

    Called by factory fixtures as they hand a resource out. Outside a test,
    for example from a session fixture's setup, there is nothing to attribute
    the resource to and the call does nothing.

    Args:
        label: A name for the resource, such as the index name.
        fingerprint: Reads the resource's current state. Returns ``None``
            while the resource does not exist, and otherwise a value that
            differs between two readings exactly when the resource was
            written in between.
    """
    item = _state.current_item
    if item is None:
        return
    _state.taken.setdefault(item.nodeid, []).append(_Handle(label, fingerprint))


def take_shared_resource(
    label: str, fingerprint: Callable[[], Hashable | None], *, instead: str
) -> None:
    """Record that the running test took a resource it shares with other tests.

    Called by factory fixtures each time they hand a shared resource out. The
    first call for a ``label`` reads the baseline, so a factory calls this
    once the resource is fully seeded. The factory's teardown then calls
    :func:`check_shared_resources`, and a difference is an error in that test.

    Args:
        label: What the resource is, unique in this process and read in the
            error messages, such as ``"index test_corpus_1a2b3c4d"``.
        fingerprint: Reads the resource's current state, as for
            :func:`take_fresh_resource`.
        instead: What a test that writes should take, named in the error a
            writer gets, such as ``"make_elasticsearch_test_index"``.

    Raises:
        SharedResourceWrittenError: An earlier test wrote to the resource. It
            no longer holds what it was seeded with, so this test would be
            reading something else.
    """
    entry = _state.shared.get(label)
    if entry is None:
        entry = _state.shared[label] = _Shared(fingerprint, fingerprint(), instead)
    elif entry.written_by is not None:
        raise SharedResourceWrittenError(
            f"The shared {label} was written by {entry.written_by}, so it no "
            "longer holds what it was seeded with. Fix that test; this one did "
            "nothing wrong."
        )
    item = _state.current_item
    if item is not None:
        taken = _state.shared_taken.setdefault(item.nodeid, [])
        if label not in taken:
            taken.append(label)


def check_shared_resources() -> None:
    """Raise if the running test left any shared resource it took changed.

    Called from the teardown of the factory fixture that handed the resources
    out. That runs after the teardown of every fixture that asked the factory
    for one, so a fixture that writes on its way out is caught too, and before
    any session-scoped teardown that deletes the resources.

    Every changed resource is marked written before raising, so each later
    test that asks for one is told which test wrote it.

    Raises:
        SharedResourceWrittenError: The test wrote to a shared resource.
    """
    item = _state.current_item
    if item is None:
        return
    nodeid = item.nodeid
    labels = _state.shared_taken.pop(nodeid, [])
    written = [
        (label, entry)
        for label in labels
        if (entry := _state.shared[label]).written_by is None
        and entry.fingerprint() != entry.baseline
    ]
    for _label, entry in written:
        entry.written_by = nodeid
    if written:
        names = ", ".join(label for label, _entry in written)
        remedies = ", ".join(sorted({entry.instead for _label, entry in written}))
        raise SharedResourceWrittenError(
            f"{nodeid} wrote to the shared {names}. A shared resource is seeded "
            "once and read by every test that asks for it, so a test that writes "
            f"takes one of its own: {remedies}. See .claude/rules/test-cost.md."
        )


def _repository_root(start: Path) -> Path:
    """The nearest directory at or above ``start`` holding ``.git``, else ``start``.

    A worktree's ``.git`` is a file, so existence is the test, not ``is_dir``.
    """
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


def _service_markers() -> dict[str, list[Any]]:
    """The skip marks that mean "this service is unreachable", by service."""
    from dataknobs_common.testing import _core

    return {
        "elasticsearch": [
            decorator.mark
            for decorator in (_core.requires_elasticsearch, _core.requires_real_elasticsearch)
            if decorator is not None
        ],
    }


def _skip_reason(report: Any) -> str | None:
    """The reason a skipped report gives, without pytest's ``Skipped: `` prefix."""
    longrepr = report.longrepr
    if isinstance(longrepr, tuple | list) and len(longrepr) == 3:
        reason = str(longrepr[2])
        return reason.removeprefix("Skipped: ")
    return None


class _Recorder:
    """Collects what each test reported and writes the record. Controller only."""

    def __init__(self, path: Path, rootpath: Path) -> None:
        self._path = path
        self._rootpath = rootpath
        self._root = _repository_root(rootpath)
        self._files: dict[str, dict[str, Any]] = {}
        self._skipped: Counter[str] = Counter()

    def _file(self, nodeid: str) -> str:
        relative = nodeid.split("::", 1)[0]
        absolute = (self._rootpath / relative).resolve()
        try:
            return absolute.relative_to(self._root.resolve()).as_posix()
        except ValueError:
            return relative

    def pytest_runtest_logreport(self, report: Any) -> None:
        for name, value in report.user_properties:
            if name == _COUNT_PROPERTY and report.when == "call":
                entry = self._files.setdefault(
                    self._file(report.nodeid), {"took": 0, "unwritten": []}
                )
                entry["took"] += value["took"]
                if value["unwritten"]:
                    entry["unwritten"].append(report.nodeid)
            elif name == _SKIP_PROPERTY and report.skipped and report.when == "setup":
                service, reason = value
                if _skip_reason(report) == reason:
                    self._skipped[service] += 1

    def pytest_unconfigure(self) -> None:
        _state.recorder_active = False

    def pytest_sessionfinish(self) -> None:
        record = {
            "version": RECORD_VERSION,
            # Sorted throughout: reports arrive in the order xdist workers finish,
            # which varies run to run, and the record must not.
            "files": {
                name: {
                    "took": entry["took"],
                    "unwritten": sorted(entry["unwritten"]),
                }
                for name, entry in sorted(self._files.items())
            },
            "service_skipped": {
                service: self._skipped[service] for service in sorted(_service_markers())
            },
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


try:
    import pytest

    def pytest_configure(config: pytest.Config) -> None:
        """Register the recorder on the controller when a record is asked for.

        Not on an xdist worker (``workerinput``), and not in any session a
        worker process starts in-process (``PYTEST_XDIST_WORKER``), and not in
        a session nested inside one that already records.
        """
        path = os.environ.get(RECORD_ENV)
        if (
            path
            and not hasattr(config, "workerinput")
            and "PYTEST_XDIST_WORKER" not in os.environ
            and not _state.recorder_active
        ):
            _state.recorder_active = True
            config.pluginmanager.register(
                _Recorder(Path(path), config.rootpath), "dataknobs-provisioning-recorder"
            )

    def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
        """Tag each test a service marker could skip, with the reason that skip gives."""
        if not recording():
            return
        markers = _service_markers()
        for item in items:
            for mark in item.iter_markers("skipif"):
                for service, service_marks in markers.items():
                    if mark in service_marks:
                        item.user_properties.append(
                            (_SKIP_PROPERTY, (service, mark.kwargs.get("reason")))
                        )

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[Any]:
        """Name the running test for :func:`take_fresh_resource`, setup to teardown."""
        # Restored rather than cleared: a session nested in-process inside a
        # test returns the outer test to its own teardown.
        outer = _state.current_item
        _state.current_item = item
        try:
            return (yield)
        finally:
            _state.current_item = outer
            _state.taken.pop(item.nodeid, None)
            _state.shared_taken.pop(item.nodeid, None)

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_call(item: pytest.Item) -> Iterator[Any]:
        """Fingerprint each fresh resource as the call begins and as it ends.

        Only the call phase is read, so seeding in setup is not counted as
        the test writing. Resources taken during the call itself were not
        provisioned for it in setup, and are not read.
        """
        if not recording():
            return (yield)
        handles = list(_state.taken.get(item.nodeid, []))
        before = [(handle, handle.fingerprint()) for handle in handles]
        try:
            return (yield)
        finally:
            if handles:
                unwritten = [
                    handle.label
                    for handle, reading in before
                    if reading is not None and handle.fingerprint() == reading
                ]
                item.user_properties.append(
                    (_COUNT_PROPERTY, {"took": len(handles), "unwritten": unwritten})
                )

except ImportError:
    # pytest not installed: there are no tests to count. take_fresh_resource
    # and recording() above remain importable and inert.
    pass

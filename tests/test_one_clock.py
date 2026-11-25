"""Shipped code reads the current UTC time in one place: ``dataknobs_common.clock``.

Every aware timestamp the library writes goes through
:func:`dataknobs_common.clock.utc_now` or :func:`~dataknobs_common.clock.now_iso`,
so :func:`~dataknobs_common.clock.use_clock` fixes all of them in a test. An
aware read of the current time written anywhere else is a stamp that seam
cannot reach: it reads wall-clock time under a fixed clock, and a test
asserting an ordering across it and a fixed stamp is quietly wrong. Before the
module existed there were 86 such lines across four packages.

**What counts.** The ways of producing an aware ``datetime`` for *now*, found
by walking each module's syntax tree, so spacing, line breaks and aliases make
no difference:

- ``now(...)`` called with a zone that is not ``None``:
  ``datetime.now(UTC)``, ``dt.datetime.now(tz=timezone.utc)``,
  ``pd.Timestamp.now("UTC")``;
- a zone attached to a naive read: ``datetime.now().astimezone(...)``,
  ``datetime.utcnow().replace(tzinfo=...)``;
- ``fromtimestamp(time.time(), <zone>)``;
- ``functools.partial(<x>.now, <zone>)``, the reference form a
  ``default_factory`` takes.

**What does not, by design.** Two populations of timestamps are outside this
seam, and both are known:

- **Naive reads** -- ``datetime.now()``, ``datetime.utcnow()``,
  ``default_factory=datetime.now``. A stamp written in local time needs a
  timezone policy for the values already stored, and routing it through
  ``utc_now()`` without that decision would change stored values.
- **Epoch floats** -- ``time.time()``. These are a different representation
  (a registry's ``registered_at``, a wizard's stage-entry time, a memory
  bank's ``created_at``), and moving one changes what a reader stored or
  compares against.

``use_clock`` does not reach either. When they move, this guard widens to
them. An elapsed time is a third case and stays out for good: it is measured
with :func:`time.monotonic`, not with any wall clock.

It reads ``git ls-files``, so a new file is seen once it is added.
"""

from __future__ import annotations

import ast
import subprocess

from tests._workspace import ROOT

#: The one file allowed to read the system clock for a timestamp.
CLOCK_MODULE = "packages/common/src/dataknobs_common/clock.py"


def _source_files() -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "--", ":(glob)packages/*/src/**/*.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert listed, "git ls-files found no package sources; the guard would pass vacuously"
    return listed


def _is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _zone_given(call: ast.Call, *, keyword: str) -> bool:
    """Whether ``call`` passes a zone that is not ``None``, positionally or by name."""
    zones = [*call.args[:1], *(k.value for k in call.keywords if k.arg == keyword)]
    return any(not _is_none(zone) for zone in zones)


def _attr_call(node: ast.AST, *names: str) -> bool:
    """Whether ``node`` is a call of a method spelled one of ``names``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in names
    )


def _is_time_time(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "time"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "time"
    )


def _aware_read(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        if func.attr == "now":
            return _zone_given(node, keyword="tz")
        if func.attr == "astimezone" and _attr_call(func.value, "now", "utcnow"):
            return True
        if func.attr == "replace" and _attr_call(func.value, "now", "utcnow"):
            return any(k.arg == "tzinfo" and not _is_none(k.value) for k in node.keywords)
        if func.attr == "fromtimestamp" and node.args and _is_time_time(node.args[0]):
            zones = [*node.args[1:2], *(k.value for k in node.keywords if k.arg == "tz")]
            return any(not _is_none(zone) for zone in zones)
    is_partial = (isinstance(func, ast.Name) and func.id == "partial") or (
        isinstance(func, ast.Attribute) and func.attr == "partial"
    )
    if is_partial and node.args:
        target = node.args[0]
        zones = [*node.args[1:2], *(k.value for k in node.keywords if k.arg == "tz")]
        return (
            isinstance(target, ast.Attribute)
            and target.attr == "now"
            and any(not _is_none(zone) for zone in zones)
        )
    return False


def _aware_reads(text: str) -> list[int]:
    return sorted(
        {
            node.lineno
            for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.Call) and _aware_read(node)
        }
    )


def test_the_clock_module_is_tracked_and_reads_the_system_clock() -> None:
    files = _source_files()
    assert CLOCK_MODULE in files, f"{CLOCK_MODULE} is not tracked (run after `git add`)"
    assert _aware_reads((ROOT / CLOCK_MODULE).read_text(encoding="utf-8"))


def test_no_other_shipped_file_reads_the_current_utc_time() -> None:
    found = [
        f"{path}:{number}"
        for path in _source_files()
        if path != CLOCK_MODULE
        for number in _aware_reads((ROOT / path).read_text(encoding="utf-8"))
    ]
    assert not found, (
        "read the current time through dataknobs_common.clock (utc_now / now_iso), "
        "so use_clock reaches the stamp:\n  " + "\n  ".join(found)
    )


def test_the_walk_catches_each_aware_spelling() -> None:
    for source in (
        "now = datetime.now(UTC)",
        "x = datetime.now(tz=timezone.utc)",
        "stamp = _datetime.now(UTC).isoformat()",
        "t = dt.datetime.now( datetime.UTC )",
        "t = datetime.now(\n    UTC,\n)",
        "t = pd.Timestamp.now('UTC')",
        "t = datetime.now().astimezone(UTC)",
        "t = datetime.now().astimezone()",
        "t = datetime.utcnow().replace(tzinfo=UTC)",
        "t = datetime.fromtimestamp(time.time(), UTC)",
        "t = datetime.fromtimestamp(time.time(), tz=UTC)",
        "f = field(default_factory=partial(datetime.now, UTC))",
        "f = functools.partial(datetime.now, tz=UTC)",
    ):
        assert _aware_reads(source), source


def test_the_walk_leaves_naive_and_epoch_reads_alone() -> None:
    for source in (
        "t = datetime.now()",
        "t = datetime.now(None)",
        "t = datetime.now(tz=None)",
        "t = datetime.utcnow()",
        "t = datetime.utcnow().replace(tzinfo=None)",
        "f = field(default_factory=datetime.now)",
        "t = time.time()",
        "t = datetime.fromtimestamp(time.time())",
        "t = datetime.fromtimestamp(stored, UTC)",
        "t = utc_now()",
        "t = time.monotonic()",
        "t = stamp.astimezone(UTC)",
    ):
        assert not _aware_reads(source), source

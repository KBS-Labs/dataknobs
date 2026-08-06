"""Guards the JSON contract between the artifact validator and its CI consumer.

``bin/package-hashes.py validate --json`` produces a verdict;
``bin/validate-quality-artifacts.sh`` consumes it and is what the quality
workflow actually runs. Nothing connects the two — the producer can add a field,
or a new pairing of fields, and the consumer keeps parsing the shape it was
written against. Both failures in this file's history are that shape:

* ``changed_scopes`` was added to the producer and never read, so a
  workspace-only change failed the gate while naming nothing that changed.
* ``warning`` was chained ahead of the verdict, so a result carrying both a
  warning and a failure reported the warning and skipped the failure. That is
  a green gate on stale artifacts, and it appeared independently in the Python
  CLI and in the shell — the same mistake twice, which is what makes it a class
  rather than an incident.

Both are checked here against the producer's *actual* emitted keys, read from
its source, so a field added later is covered without an edit to this file.
"""

from __future__ import annotations

import ast
import re

from tests._workspace import ROOT

PRODUCER = ROOT / "bin" / "package-hashes.py"
CONSUMER = ROOT / "bin" / "validate-quality-artifacts.sh"

#: Emitted for completeness but deliberately not read from this payload. The
#: consumer greps ``overall_status`` out of quality-summary.json directly, a few
#: checks further down, and ``status_ok`` is that same value already reduced to
#: a boolean — both are folded into ``valid`` before the consumer sees them, so
#: reading them here would report the same fact twice.
NOT_CONSUMED_BY_DESIGN = frozenset({"status_ok", "overall_status"})


def _emitted_keys() -> set[str]:
    """Every key ``validate_artifacts`` can put in its result.

    Taken from the source rather than by calling it: the early returns are the
    cases that matter most here, and reaching them means standing up missing or
    corrupt artifacts on disk.
    """
    tree = ast.parse(PRODUCER.read_text(encoding="utf-8"))
    func = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "validate_artifacts"
        ),
        None,
    )
    assert func is not None, f"validate_artifacts not found in {PRODUCER.name}"

    keys: set[str] = set()
    for node in ast.walk(func):
        # Dict literals: the early returns and the main result.
        if isinstance(node, ast.Dict):
            keys |= {k.value for k in node.keys if isinstance(k, ast.Constant)}
        # Later additions: result["warning"] = ...
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.value, ast.Name) and isinstance(node.slice.value, str):
                keys.add(node.slice.value)
    return keys


def test_producer_emits_the_keys_this_guard_expects():
    """Non-vacuity: the AST walk must actually find the payload.

    A renamed function or a restructured return would leave the extraction
    empty, and every check below would then pass by comparing nothing.
    """
    keys = _emitted_keys()
    assert keys, "no result keys extracted — the producer's shape changed"
    missing = {"valid", "warning", "error", "changed_packages"} - keys
    assert not missing, f"expected result keys not found: {sorted(missing)}"


def test_consumer_reads_every_field_the_producer_emits():
    """A field the consumer never reads is a diagnostic that reaches no one.

    This is the ``changed_scopes`` failure exactly: the producer knew which
    workspace scope had changed, the gate failed, and the report named nothing,
    because the one script that prints the reason had never been taught the
    field existed.
    """
    consumer = CONSUMER.read_text(encoding="utf-8")
    unread = sorted(
        key
        for key in _emitted_keys() - NOT_CONSUMED_BY_DESIGN
        if f"'{key}'" not in consumer and f'"{key}"' not in consumer
    )
    assert not unread, (
        f"{CONSUMER.name} never reads {unread}, so that detail is computed and "
        "then dropped. Print it in the failure branch, or add it to "
        "NOT_CONSUMED_BY_DESIGN with the reason."
    )


def test_consumer_does_not_let_a_warning_shadow_the_verdict():
    """``warning`` must be printed additively, never as a branch before the verdict.

    A warning says what could *not* be checked. It carries no claim about what
    was checked, so a chain that tests it first will, on any result carrying
    both, print the warning and never reach the failure. The producer emits
    exactly that pairing whenever artifacts predate workspace hashing and a
    package is also dirty.
    """
    lines = CONSUMER.read_text(encoding="utf-8").splitlines()
    branches = [ln.strip() for ln in lines if "HASH_WARNING" in ln and re.match(r"^\s*(el)?if ", ln)]
    assert branches, (
        f"{CONSUMER.name} no longer branches on HASH_WARNING at all — if the "
        "variable was renamed, update this guard rather than deleting it"
    )
    chained = [b for b in branches if b.startswith("elif")]
    assert not chained, (
        "The warning is tested as 'elif', so a result carrying both a warning "
        f"and a failure reports only the warning: {chained}. Print it in its "
        "own 'if' ahead of the verdict instead."
    )

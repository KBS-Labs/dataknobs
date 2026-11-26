"""A test that reaches a real Elasticsearch node takes its index name from the plugin.

``make_elasticsearch_test_index`` hands out a unique name, deletes the index
on teardown, and reports it to the provisioning count, which the quality
contract holds under a ceiling. A test that names its own index gets none of
that:

* **A fixed name is shared by every run.** Two worktrees running the gate at
  once write into one index, and the session sweep deletes ``test_*`` indices
  older than its age threshold, so it can remove a fixed-name index another
  run is holding.
* **The count cannot see it.** An index the plugin did not hand out is never
  fingerprinted, so a test that names its own goes round the ceiling.

The population is every test file that references ``requires_real_elasticsearch``,
the marker for "this test reaches a real node". Files that only build a config
and never connect carry no such marker and are not read. The plugin's own tests
in ``packages/common/tests`` use ``requires_elasticsearch`` and make names by
hand on purpose, since they test the machinery that hands names out.

What is refused is an index name *spelled* in the test, as a string literal or
an f-string, in any of the three places a name is given:

* ``{"index": "..."}`` in a config dict,
* ``index="..."`` / ``index_name="..."`` as a keyword,
* ``index = "..."`` / ``index_name = "..."`` assigned to a variable.

A name derived from a plugin-issued one, such as ``config["index"] + "_down"``,
is an expression over that name and passes.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests._workspace import ROOT, tracked_and_new_files

MARKER = "requires_real_elasticsearch"
NAMES = frozenset({"index", "index_name"})


def _population() -> list[str]:
    files = [
        name
        for name in tracked_and_new_files()
        if name.startswith("packages/")
        and "/tests/" in name
        and name.endswith(".py")
        and (ROOT / name).is_file()
    ]
    return sorted(name for name in files if MARKER in (ROOT / name).read_text(encoding="utf-8"))


def _spelled(node: ast.AST | None) -> bool:
    """A name written out in the test: a str literal, or an f-string."""
    return isinstance(node, ast.JoinedStr) or (
        isinstance(node, ast.Constant) and isinstance(node.value, str)
    )


def hand_made_index_names(source: str) -> list[int]:
    """The lines on which ``source`` spells an index name instead of taking one."""
    lines: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value in NAMES and _spelled(value):
                    lines.append(value.lineno)
        elif isinstance(node, ast.keyword):
            if node.arg in NAMES and _spelled(node.value):
                lines.append(node.value.lineno)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in NAMES and _spelled(node.value):
                    lines.append(node.lineno)
    return sorted(lines)


def test_the_population_is_not_empty() -> None:
    """A guard over nothing passes on any tree."""
    assert len(_population()) > 5, _population()


def test_no_test_that_reaches_a_node_names_its_own_index() -> None:
    offenders = {
        name: lines
        for name in _population()
        if (lines := hand_made_index_names((ROOT / name).read_text(encoding="utf-8")))
    }
    assert not offenders, (
        "These test files reach a real Elasticsearch node and spell an index name "
        "themselves (file: lines):\n"
        + "\n".join(f"  {name}: {lines}" for name, lines in sorted(offenders.items()))
        + "\nTake the name from make_elasticsearch_test_index, which makes it unique, "
        "deletes the index afterwards, and reports it to the provisioning count. "
        "A name derived from that one (config['index'] + '_suffix') is fine."
    )


def test_each_spelling_is_recognised_and_a_derived_name_is_not() -> None:
    """The detector itself, on the three spellings and the one allowed shape."""
    source = Path(__file__).read_text(encoding="utf-8")
    assert hand_made_index_names(source) == []
    spelled = """
a = {"index": "test_fixed"}
b = {"index": f"test_{suffix}"}
connect(index="test_kw")
make(index_name=f"test_{n}")
index_name = f"test_{n}"
index = "test_plain"
"""
    assert hand_made_index_names(spelled) == [2, 3, 4, 5, 6, 7]
    derived = """
up = config["index"] + "_down"
a = {"index": config["index"]}
b = {**config, "index": alias}
admin.indices.create(index=config["index"] + suffix)
other = {"host": "localhost"}
"""
    assert hand_made_index_names(derived) == []

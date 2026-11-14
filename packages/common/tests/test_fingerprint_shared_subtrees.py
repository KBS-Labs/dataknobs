"""A digest costs what a value holds, not how many paths reach it.

A container reachable by two paths -- what a YAML anchor builds -- is one
container, and a digest reads it once. Before, the encoding was spelled out
once per path, so a structure sharing a subtree at each level cost time
exponential in its depth, and an authored vocabulary paid it at load, since a
mapping source digests its entities when it is built.

What this file pins:

* a shared container is read once, through ``fingerprint_value``, through
  ``fingerprint()`` and through the loader door;
* sharing never changes a digest: a shared subtree and an equal copy of it
  digest alike, for every kind of container;
* a cycle is still refused, including one a hook's answer closes;
* the memo cannot hand one value's digest to another whose ``id`` it reuses.
"""

from __future__ import annotations

import copy
import subprocess
import sys
import textwrap
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

from dataknobs_common import fingerprint_value
from dataknobs_common.structured_config import StructuredConfig


class CountingMapping(Mapping[str, Any]):
    """A real mapping that counts how often its pairs are read."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self.reads = 0

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def items(self) -> Any:
        self.reads += 1
        return self._data.items()


def _shared_chain(leaf: Any, depth: int) -> list[Any]:
    """``depth`` levels of ``[v, v]`` over ``leaf``: 2**depth paths to it."""
    value: Any = leaf
    for _ in range(depth):
        value = [value, value]
    return value


@dataclass(frozen=True)
class Holder(StructuredConfig):
    payload: Any = None


class Opaque:
    """A value nothing identifies without a hook."""

    def __init__(self, n: int) -> None:
        self.n = n


@dataclass(frozen=True)
class Rewrap(StructuredConfig):
    """Answers a fresh config holding the very value it was asked about.

    The nested config turns the hook back on, so the value comes round again:
    a cycle through the hook, which no container closes.
    """

    endpoint: Any = None

    def _fingerprint_value(self, value: Any, path: str) -> Any:
        return Rewrap(endpoint=value)


@dataclass(frozen=True)
class FreshAnswers(StructuredConfig):
    """Answers a new list per value, so each answer is garbage once encoded."""

    items: tuple[Any, ...] = field(default_factory=tuple)

    def _fingerprint_value(self, value: Any, path: str) -> Any:
        if isinstance(value, Opaque):
            return [f"opaque-{value.n}"]
        return super()._fingerprint_value(value, path)


# --------------------------------------------------------------------------
# A shared container is read once
# --------------------------------------------------------------------------


def test_fingerprint_value_reads_a_shared_container_once() -> None:
    leaf = CountingMapping({"k": 1})
    fingerprint_value(_shared_chain(leaf, 12))
    assert leaf.reads == 1  # 4,096 paths reach it


def test_fingerprint_reads_a_shared_container_once() -> None:
    leaf = CountingMapping({"k": 1})
    Holder(payload=_shared_chain(leaf, 12)).fingerprint()
    assert leaf.reads == 1


def test_a_document_nesting_an_anchor_loads(tmp_path: Any) -> None:
    """The door: a YAML anchor referenced twice per level, forty levels deep.

    Run in a child process with a bound, because 2**40 paths do not fail,
    they never finish. The bound is against a hang, not a timing: a linear
    digest of this document takes milliseconds.
    """
    levels = ["      l0: &a0 [1]"] + [
        f"      l{i}: &a{i} [*a{i - 1}, *a{i - 1}]" for i in range(1, 41)
    ]
    document = tmp_path / "vocabulary.yaml"
    document.write_text(
        textwrap.dedent(
            """\
            id: catalog
            entity_types:
              - id: product
            entities:
              - id: e-1
                type: product
                name: Beagle
                metadata:
            """
        )
        + "\n".join(levels)
        + "\n"
    )
    script = (
        "import sys\n"
        "from dataknobs_common.ontology import load_ontology\n"
        "print(load_ontology(__import__('pathlib').Path(sys.argv[1])).entities.revision())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(document)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert result.stdout.strip().startswith("1.0@")


# --------------------------------------------------------------------------
# Sharing never changes a digest
# --------------------------------------------------------------------------


@dataclass
class Plain:
    columns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Inner(StructuredConfig):
    names: tuple[str, ...] = ()


@pytest.mark.parametrize(
    "shared",
    [
        ["a", ["b"]],
        ("a", ("b",)),
        {"k": {"j": 1}},
        frozenset({"a", "b"}),
        Plain(columns=["x", "y"]),
        Inner(names=("x", "y")),
    ],
    ids=["list", "tuple", "mapping", "frozenset", "dataclass", "config"],
)
def test_a_shared_subtree_digests_as_its_copy(shared: Any) -> None:
    """A node that depended on identity would part these two."""
    assert fingerprint_value({"a": shared, "b": shared}) == fingerprint_value(
        {"a": shared, "b": copy.deepcopy(shared)}
    )
    assert (
        Holder(payload=[shared, shared]).fingerprint()
        == Holder(payload=[shared, copy.deepcopy(shared)]).fingerprint()
    )


def test_set_members_are_ordered_by_value() -> None:
    one = frozenset([("a", 1), ("b", 2)])
    other = frozenset([("b", 2), ("a", 1)])
    assert fingerprint_value(one) == fingerprint_value(other)
    assert fingerprint_value(one) != fingerprint_value(frozenset([("a", 1), ("b", 3)]))


# --------------------------------------------------------------------------
# A cycle is still refused
# --------------------------------------------------------------------------


def test_a_shared_container_that_holds_itself_is_refused() -> None:
    looped: list[Any] = []
    looped.append(looped)
    with pytest.raises(ValueError, match="holds itself"):
        fingerprint_value([looped, looped])


def test_a_cycle_closed_by_a_hooks_answer_is_refused() -> None:
    """No container closes this cycle, so only the hook branch can see it."""
    with pytest.raises(ValueError, match=r"Rewrap\.endpoint.*holds itself"):
        Rewrap(endpoint=Opaque(1)).fingerprint()


# --------------------------------------------------------------------------
# The memo keeps what it is keyed on alive
# --------------------------------------------------------------------------


def test_a_fresh_answer_never_takes_an_earlier_ones_digest() -> None:
    """Each answer is garbage once encoded, and CPython reuses its ``id`` at once.

    A memo keyed on ``id`` that let them go would hand every later answer the
    first one's digest, and these two configs would collide.
    """
    distinct = tuple(Opaque(n) for n in range(200))
    repeated = (distinct[0],) * 200
    assert FreshAnswers(items=distinct).fingerprint() != FreshAnswers(items=repeated).fingerprint()

"""``copy_structure`` -- containers rebuilt, leaves shared.

The three properties that decide when this is the right copy, each
asserted against the alternative it is chosen over: it isolates where
``dict()`` does not, and it shares where ``copy.deepcopy`` would not.
"""

import copy
import threading
from typing import Any

import pytest

from dataknobs_common import copy_structure


class TestContainersAreRebuilt:
    """The half ``dict()`` does not give."""

    def test_nested_dicts_are_not_the_source_objects(self) -> None:
        source = {"schema": {"type": "object"}, "prompt": "Name?"}

        handed_out = copy_structure(source)
        handed_out["schema"]["type"] = "array"

        assert source["schema"]["type"] == "object"

    def test_a_shallow_copy_would_not_have_held(self) -> None:
        """The comparison that makes the choice legible.

        Pinned rather than described, because "returns a copy" reads as
        this guarantee and a shallow copy does not provide it.
        """
        source: dict[str, Any] = {"schema": {"type": "object"}}

        shallow = dict(source)
        shallow["schema"]["type"] = "array"

        assert source["schema"]["type"] == "array", (
            "precondition: dict() leaves the nested container aliased"
        )

    def test_nested_lists_are_rebuilt(self) -> None:
        source = {"transitions": [{"target": "next"}]}

        handed_out = copy_structure(source)
        handed_out["transitions"].append({"target": "other"})
        handed_out["transitions"][0]["target"] = "changed"

        assert source["transitions"] == [{"target": "next"}]

    def test_a_bare_list_is_rebuilt_too(self) -> None:
        source = [[1], [2]]

        handed_out = copy_structure(source)
        handed_out[0].append(99)

        assert source == [[1], [2]]


class TestLeavesArePassedThrough:
    """The half ``copy.deepcopy`` does not give -- and the reason why."""

    def test_a_live_object_is_not_duplicated(self) -> None:
        """The defect the pass-through exists to prevent.

        A structure assembled in Python may hold a live object. Copying
        it would hand its owner a second one, silently.
        """
        lock = threading.Lock()
        source = {"resource": {"lock": lock}}

        handed_out = copy_structure(source)

        assert handed_out["resource"] is not source["resource"]
        assert handed_out["resource"]["lock"] is lock

    def test_an_uncopyable_value_does_not_raise(self) -> None:
        """``deepcopy`` raising out of an ordinary read is the hazard."""

        class Uncopyable:
            def __deepcopy__(self, memo: dict[int, Any]) -> Any:
                raise TypeError("this object cannot be copied")

        value = Uncopyable()
        source = {"resource": value}

        with pytest.raises(TypeError):
            copy.deepcopy(source)

        assert copy_structure(source)["resource"] is value


class TestTheMemo:
    """Kept from ``deepcopy`` for the reasons ``deepcopy`` keeps it."""

    def test_a_self_referential_structure_terminates(self) -> None:
        source: dict[str, Any] = {"name": "root"}
        source["self"] = source

        handed_out = copy_structure(source)

        assert handed_out["self"] is handed_out
        assert handed_out is not source

    def test_a_shared_subtree_stays_shared(self) -> None:
        """Sharing on the way in is sharing on the way out."""
        subtree = {"x": 1}
        source = {"first": subtree, "second": subtree}

        handed_out = copy_structure(source)

        assert handed_out["first"] is handed_out["second"]
        assert handed_out["first"] is not subtree

    def test_one_memo_across_several_calls_preserves_sharing(self) -> None:
        """What a caller assembling one hand-out from several values needs."""
        subtree = {"x": 1}
        seen: dict[int, Any] = {}

        first = copy_structure({"a": subtree}, seen)
        second = copy_structure({"b": subtree}, seen)

        assert first["a"] is second["b"]
        assert first["a"] is not subtree

    def test_a_freed_source_cannot_have_its_id_answered_for(self) -> None:
        """The memo is keyed on ``id()``, and ids are reused.

        Both sources here are temporaries: the first is freed when the
        call returns, and CPython hands its address to the second. A memo
        holding only ``id -> copy`` then answers for an unrelated object
        and returns the first call's result -- observed, not theorised.
        ``deepcopy`` avoids this by keeping every source alive, and so
        does this.
        """
        subtree = {"x": 1}
        seen: dict[int, Any] = {}

        first = copy_structure({"first_key": subtree}, seen)
        second = copy_structure({"second_key": subtree}, seen)

        assert first is not second
        assert "second_key" in second, (
            "the memo answered for a freed source whose id was reused"
        )
        assert first["first_key"] is second["second_key"], (
            "the shared subtree is still shared"
        )

    def test_separate_calls_without_a_memo_do_not_share(self) -> None:
        """The contrast that makes passing one memo a decision."""
        subtree = {"x": 1}

        first = copy_structure({"a": subtree})
        second = copy_structure({"b": subtree})

        assert first["a"] is not second["b"]


class TestNonContainers:
    """Everything else is returned as itself."""

    @pytest.mark.parametrize(
        "value", ["text", 42, 3.5, True, None, (1, 2), frozenset({1})]
    )
    def test_returned_by_identity(self, value: Any) -> None:
        assert copy_structure(value) is value

    def test_a_tuple_is_not_rebuilt(self) -> None:
        """Documented boundary: only ``dict`` and ``list`` are rebuilt.

        A tuple is immutable, so identity is not a hazard -- but a dict
        *inside* one is not reached, and a caller relying on isolation
        through a tuple needs to know that.
        """
        inner = {"x": 1}
        source = {"pair": (inner,)}

        handed_out = copy_structure(source)

        assert handed_out["pair"][0] is inner

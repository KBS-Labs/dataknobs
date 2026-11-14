"""``copy_structure`` and ``freeze_structure`` -- containers rebuilt, leaves shared.

The three properties that decide when ``copy_structure`` is the right copy,
each asserted against the alternative it is chosen over: it isolates where
``dict()`` does not, and it shares where ``copy.deepcopy`` would not.

``freeze_structure`` is the same walk with immutable results, and
``FrozenMapping`` is the mapping it builds; their tests follow.
"""

import copy
import pickle
import threading
import types
from collections.abc import Hashable
from typing import Any

import pytest

from dataknobs_common import FrozenMapping, copy_structure, freeze_structure


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
        assert "second_key" in second, "the memo answered for a freed source whose id was reused"
        assert first["first_key"] is second["second_key"], "the shared subtree is still shared"

    def test_separate_calls_without_a_memo_do_not_share(self) -> None:
        """The contrast that makes passing one memo a decision."""
        subtree = {"x": 1}

        first = copy_structure({"a": subtree})
        second = copy_structure({"b": subtree})

        assert first["a"] is not second["b"]


class TestNonContainers:
    """Everything else is returned as itself."""

    @pytest.mark.parametrize("value", ["text", 42, 3.5, True, None, (1, 2), frozenset({1})])
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

    def test_a_mutable_set_is_not_rebuilt(self) -> None:
        """The pass-through that is not safe by immutability.

        ``tuple`` is the example the docs lead with, and its safety comes
        from being immutable. A ``set`` passes through on the same rule
        while being mutable, so a write through the result reaches the
        source -- the one case where "leaves are shared" costs isolation
        rather than merely declining to buy more of it.
        """
        source = {"tags": {"a", "b"}}

        handed_out = copy_structure(source)
        assert handed_out["tags"] is source["tags"]

        handed_out["tags"].add("added")
        assert "added" in source["tags"]


class TestFrozenMapping:
    """A read-only mapping that keeps what a plain dict can do."""

    def test_it_refuses_every_write(self) -> None:
        frozen = FrozenMapping({"k": 1})
        with pytest.raises(TypeError):
            frozen["k"] = 2  # type: ignore[index]
        with pytest.raises(TypeError):
            del frozen["k"]  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            frozen.update({"k": 2})  # type: ignore[attr-defined]
        assert frozen == {"k": 1}

    def test_it_equals_a_dict_with_the_same_items_from_either_side(self) -> None:
        assert FrozenMapping({"k": 1}) == {"k": 1}
        assert FrozenMapping({"k": 1}) == {"k": 1}
        assert FrozenMapping({"k": 1}) != {"k": 2}

    def test_it_answers_hashable_honestly(self) -> None:
        frozen = FrozenMapping({"k": 1})
        assert not isinstance(frozen, Hashable)
        with pytest.raises(TypeError):
            hash(frozen)

    def test_it_freezes_what_it_is_handed_all_the_way_down(self) -> None:
        frozen = FrozenMapping({"tags": ["a"], "inner": {"k": [1]}})
        assert frozen["tags"] == ("a",)
        assert isinstance(frozen["inner"], FrozenMapping)
        assert frozen["inner"]["k"] == (1,)

    def test_it_accepts_pairs_as_dict_does(self) -> None:
        assert FrozenMapping([("a", 1), ("b", 2)]) == {"a": 1, "b": 2}

    def test_the_callers_dict_does_not_reach_it(self) -> None:
        source = {"k": 1}
        frozen = FrozenMapping(source)
        source["k"] = 2
        assert frozen["k"] == 1

    def test_it_survives_pickle_and_deepcopy_where_a_proxy_does_not(self) -> None:
        """The property that decided against ``types.MappingProxyType``."""
        frozen = FrozenMapping({"k": {"inner": [1]}})

        for twin in (pickle.loads(pickle.dumps(frozen)), copy.deepcopy(frozen)):
            assert twin == frozen
            assert isinstance(twin, FrozenMapping)
            assert isinstance(twin["k"], FrozenMapping)

        proxy = types.MappingProxyType({"k": 1})
        with pytest.raises(TypeError):
            pickle.dumps(proxy)


class TestFreezeStructure:
    """``copy_structure``'s walk, with immutable results."""

    def test_nested_containers_become_immutable(self) -> None:
        frozen = freeze_structure({"d": {"k": 1}, "l": [1, [2]], "s": {3}, "t": (4, [5])})

        assert isinstance(frozen, FrozenMapping)
        assert isinstance(frozen["d"], FrozenMapping)
        assert frozen["l"] == (1, (2,))
        assert frozen["s"] == frozenset({3})
        assert frozen["t"] == (4, (5,))

    def test_a_live_object_passes_through_by_identity(self) -> None:
        """For ``copy_structure``'s reason: duplicating one gives its owner a second."""
        lock = threading.Lock()
        assert freeze_structure({"lock": lock})["lock"] is lock

    @pytest.mark.parametrize(
        "value",
        [FrozenMapping({"k": (1,)}), frozenset({1}), (1, "a", (2,)), "text", 3],
    )
    def test_an_already_frozen_value_comes_back_as_itself(self, value: Any) -> None:
        assert freeze_structure(value) is value

    def test_a_tuple_holding_a_list_is_rebuilt(self) -> None:
        source = (1, [2])
        assert freeze_structure(source) == (1, (2,))
        assert freeze_structure(source) is not source

    def test_a_cycle_is_refused_where_copy_structure_reproduces_it(self) -> None:
        source: dict[str, Any] = {"name": "loop"}
        source["self"] = source

        copied = copy_structure(source)
        assert copied["self"] is copied

        with pytest.raises(ValueError, match=r"value\['self'\] holds itself"):
            freeze_structure(source)

    def test_a_cycle_through_a_list_names_the_index(self) -> None:
        source: list[Any] = [1]
        source.append({"back": source})
        with pytest.raises(ValueError, match=r"value\[1\]\['back'\] holds itself"):
            freeze_structure(source)

    def test_a_shared_subtree_is_not_a_cycle(self) -> None:
        shared = [1]
        frozen = freeze_structure({"a": shared, "b": shared})
        assert frozen == {"a": (1,), "b": (1,)}

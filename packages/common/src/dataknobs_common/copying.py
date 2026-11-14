# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Copy or freeze nested containers without duplicating what they hold.

Two functions over one walk, for the case that sits between ``dict(value)``
and ``copy.deepcopy(value)``: the *structure* is rebuilt while the *values*
stay whatever the source had.

* :func:`copy_structure` rebuilds it mutable, so a hand-out is the caller's
  own to change.
* :func:`freeze_structure` rebuilds it immutable, so a value held by
  something that indexed it cannot be changed under that index. Its mappings
  are :class:`FrozenMapping`.

See :func:`copy_structure` for when each of the three copies is right.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, TypeVar

__all__ = ["FrozenMapping", "copy_structure", "freeze_structure"]

_KT = TypeVar("_KT")
_VT_co = TypeVar("_VT_co", covariant=True)

# The keep-alive list is filed under the memo's *own* id, which is what
# ``copy.deepcopy`` does. It cannot collide with the id of anything being
# copied -- the memo is not part of the structure -- and it keeps the memo
# honestly typed as ``dict[int, Any]`` rather than widening it to admit a
# sentinel key of another type.


def copy_structure(value: Any, seen: dict[int, Any] | None = None) -> Any:
    """Rebuild nested dicts and lists; pass every other value through.

    A shallow ``dict(value)`` isolates only the top level: every nested
    container in the result is still the source's own object, so a
    consumer adjusting a nested section writes back into a structure that
    outlives the hand-out. That is the defect this function exists to
    prevent, and it is the reason a "returns a copy" docstring is so
    often wrong about what it returns.

    **``copy.deepcopy`` would overshoot.** A structure assembled in Python
    may hold a live object -- a connection pool, a prebuilt provider, a
    lock -- and duplicating one silently gives its owner a second pool,
    while a value that cannot be pickled raises out of what is meant to
    be an ordinary read. Neither is a risk the aliasing this copy
    prevents justifies taking. It is also cheaper: on a wizard stage
    config, roughly half the cost of ``deepcopy``.

    So the boundary is: **containers are rebuilt, leaves are shared.**
    Mutating the returned structure never reaches the source; mutating a
    leaf object reachable from it still does, because that object was
    never copied. When the leaves are immutable -- the ordinary case for
    configuration loaded from YAML or JSON -- the result is a full
    isolation at a fraction of the price.

    A ``set`` is the leaf worth naming, because it is the mutable one. It
    passes through like any other non-container, so ``add()`` on a set
    reachable from the result reaches the source. Nothing nests below it
    -- a set cannot hold a dict or a list -- but the set itself is shared,
    and a caller who will be adding to one should rebuild it.

    ``seen`` is the memo, kept from ``deepcopy`` for the same reason
    ``deepcopy`` has one. A container reached twice is copied once and
    the same copy used both times, so a structure that refers back to
    itself terminates instead of raising ``RecursionError``, and one that
    merely shares a subtree between two keys keeps sharing it. A caller
    assembling a single hand-out from several values should pass one memo
    to each call, so the result's sharing reflects the source's; a caller
    copying one value omits it.

    The memo also holds a reference to every source it has seen, which
    ``deepcopy`` does through ``_keep_alive`` and for the same reason: it
    is keyed on ``id()``, and an id is only unique among *live* objects.
    Without this, a source freed between two calls sharing one memo can
    have its id reused by the next -- and the memo then answers for an
    unrelated object, returning the earlier call's copy. Not theoretical:
    two successive calls passing a temporary dict literal reproduce it.

    Args:
        value: The value to copy. Any type; only ``dict`` and ``list``
            are rebuilt.
        seen: Optional memo mapping ``id()`` to the copy already made for
            it. Pass the same dict across several calls that build one
            hand-out; omit it otherwise.

    Returns:
        A structure whose dicts and lists are new objects and whose other
        values are the originals.

    Example:
        ```python
        from dataknobs_common import copy_structure

        source = {"schema": {"type": "object"}, "prompt": "Name?"}
        handed_out = copy_structure(source)

        handed_out["schema"]["type"] = "array"
        assert source["schema"]["type"] == "object"   # not reached

        # dict() would not have held:
        shallow = dict(source)
        shallow["schema"]["type"] = "array"
        assert source["schema"]["type"] == "array"    # reached
        ```
    """
    if seen is None:
        seen = {}
    marker = id(value)
    if marker in seen:
        return seen[marker]

    if isinstance(value, (dict, list)):
        # Hold the source so its id cannot be reused while the memo
        # lives; see the note on ``seen`` above.
        seen.setdefault(id(seen), []).append(value)

    if isinstance(value, dict):
        copied_dict: dict[Any, Any] = {}
        # Registered before recursing, so a self-reference finds it.
        seen[marker] = copied_dict
        for key, item in value.items():
            copied_dict[key] = copy_structure(item, seen)
        return copied_dict
    if isinstance(value, list):
        copied_list: list[Any] = []
        seen[marker] = copied_list
        for item in value:
            copied_list.append(copy_structure(item, seen))
        return copied_list
    return value


class FrozenMapping(Mapping[_KT, _VT_co]):
    """A read-only mapping whose nested containers are frozen too.

    Built from any mapping, or from what ``dict()`` accepts. Construction
    runs :func:`freeze_structure` over every value, so a dict nested inside
    becomes a ``FrozenMapping`` and a list a tuple: nothing reachable through
    one can be changed in place, which is the property a value indexed at
    construction needs.

    **Why not** :class:`types.MappingProxyType`. A proxy is shallow -- the
    values under it stay writable -- and it cannot be pickled or
    ``deepcopy``'d, which every value holding a plain dict can be. This
    pickles and deep-copies through :meth:`__reduce__`, as the dict it holds
    does.

    **Compared as a mapping, and therefore unhashable.** It equals any
    mapping with the same items, a plain ``dict`` included, so a type that
    held a dict compares exactly as it did. Equality between mappings is not
    something a hash over the items can agree with when the items are
    unhashable, so ``__hash__`` is declared ``None`` and
    :class:`collections.abc.Hashable` answers no rather than the call raising.

    Example:
        ```python
        from dataknobs_common import FrozenMapping

        meta = FrozenMapping({"tags": ["hound"], "size": "m"})
        assert meta == {"tags": ("hound",), "size": "m"}
        assert meta["tags"] == ("hound",)   # the list came back a tuple
        ```
    """

    __slots__ = ("_data",)

    _data: dict[_KT, _VT_co]

    def __init__(self, items: Any = (), /) -> None:
        """Freeze ``items``: a mapping, or an iterable of key-value pairs.

        Raises:
            ValueError: A container under ``items`` holds itself.
        """
        source = items if isinstance(items, Mapping) else dict(items)
        self._data = {
            key: _freeze(value, f"value[{key!r}]", set()) for key, value in source.items()
        }

    @classmethod
    def _adopt(cls, data: dict[Any, Any]) -> FrozenMapping[Any, Any]:
        """Wrap ``data``, whose values are already frozen, without walking it again."""
        frozen: FrozenMapping[Any, Any] = cls.__new__(cls)
        frozen._data = data
        return frozen

    def __getitem__(self, key: _KT) -> _VT_co:
        return self._data[key]

    def __iter__(self) -> Iterator[_KT]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self)._adopt, (self._data,))


# Assigned after the class body for the reason ``StructuredConfig.__hash__`` is:
# mypy reads a body ``__hash__ = None`` as an override of ``object.__hash__``.
FrozenMapping.__hash__ = None  # type: ignore[assignment]


def freeze_structure(value: Any) -> Any:
    """Rebuild nested containers immutable; pass every other value through.

    :func:`copy_structure`'s walk with immutable results:

    ======================== ==========================================
    Handed                   Returned
    ======================== ==========================================
    a ``Mapping``            a :class:`FrozenMapping` of frozen values
    a ``list`` or ``tuple``  a ``tuple`` of frozen values
    a ``set``                a ``frozenset``
    anything else            the same object
    ======================== ==========================================

    **Leaves pass through, for** :func:`copy_structure`'s **reason**: a live
    object duplicated silently gives its owner a second one. So a mutable
    object that is not one of those containers stays mutable, and this is a
    freeze of the *structure* rather than of everything reachable.

    **An already-frozen value comes back as itself**: a ``FrozenMapping``
    (whose construction froze it), a ``frozenset``, and a tuple whose items
    are all already frozen. So freezing twice copies nothing.

    **A cycle is refused, not reproduced.** This is where the walk departs
    from :func:`copy_structure`, whose memo rebuilds a self-referring
    structure the way ``copy.deepcopy`` does. A mutable copy can be made to
    hold itself; a tuple or a ``FrozenMapping`` cannot, because each is
    complete before anything can hold it. A subtree merely *shared* between
    two keys is frozen once per occurrence, which loses the sharing and
    nothing else, since neither copy can change.

    Args:
        value: The value to freeze. Any type.

    Returns:
        The value with its mappings, lists, tuples and sets rebuilt
        immutable, and every other value the original.

    Raises:
        ValueError: A container under ``value`` holds itself. The message
            names where.

    Example:
        ```python
        from dataknobs_common import FrozenMapping, freeze_structure

        frozen = freeze_structure({"schema": {"required": ["name"]}})
        assert isinstance(frozen, FrozenMapping)
        assert frozen["schema"]["required"] == ("name",)
        ```
    """
    return _freeze(value, "value", set())


def _freeze(value: Any, path: str, active: set[int]) -> Any:
    """:func:`freeze_structure`'s walk, carrying the path and the open containers.

    ``active`` holds the ids of the containers *on the current path*, not of
    every container seen, which is what tells a cycle from a shared subtree:
    the ids are removed on the way back out. Holding each container on the
    path keeps its id from being reused while it is there.
    """
    if isinstance(value, (FrozenMapping, frozenset)):
        return value
    if not isinstance(value, (Mapping, list, tuple, set)):
        return value
    marker = id(value)
    if marker in active:
        raise ValueError(f"{path} holds itself, so it cannot be frozen.")
    active.add(marker)
    try:
        if isinstance(value, Mapping):
            return FrozenMapping._adopt(
                {key: _freeze(item, f"{path}[{key!r}]", active) for key, item in value.items()}
            )
        if isinstance(value, set):
            return frozenset(value)
        items = tuple(_freeze(item, f"{path}[{index}]", active) for index, item in enumerate(value))
        if isinstance(value, tuple) and all(a is b for a, b in zip(items, value, strict=True)):
            return value
        return items
    finally:
        active.discard(marker)

# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Copy, freeze or thaw nested containers without duplicating what they hold.

Three functions for the case that sits between ``dict(value)`` and
``copy.deepcopy(value)``: the *structure* is rebuilt while the *values* stay
whatever the source had. Each walks the containers it rebuilds and passes
every other value through.

* :func:`copy_structure` rebuilds dicts and lists mutable, so a hand-out is
  the caller's own to change.
* :func:`freeze_structure` rebuilds every mapping, sequence and set
  immutable, so a value held by something that indexed it cannot be changed
  under that index. Its mappings are :class:`FrozenMapping`.
* :func:`thaw_structure` is the inverse: plain dicts, lists and sets again,
  for a caller who needs a frozen value as JSON, or to change a copy of it.

See :func:`copy_structure` for when each of the three copies is right.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, TypeVar

__all__ = ["FrozenMapping", "copy_structure", "freeze_structure", "thaw_structure"]

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
    becomes a ``FrozenMapping`` and a list a tuple: no container reachable
    through one can be changed in place, which is the property a value
    indexed at construction needs. A leaf that is itself mutable -- a
    ``bytearray``, an object -- is held as given; see
    :func:`freeze_structure`.

    **Why not** :class:`types.MappingProxyType`. A proxy is shallow -- the
    values under it stay writable -- and it cannot be pickled or
    ``deepcopy``'d, which every value holding a plain dict can be. This
    pickles and deep-copies through :meth:`__reduce__`, as the dict it holds
    does.

    **Compared as a mapping, and therefore unhashable.** It equals any
    mapping with the same items, a plain ``dict`` included. Its items are
    frozen, so it equals a dict whose items are the same *frozen* values: a
    dict holding no list compares as it did, and one holding ``[1]`` does
    not, because the frozen side holds ``(1,)``. Equality between mappings is
    not something a hash over the items can agree with when the items are
    unhashable, so ``__hash__`` is declared ``None`` and
    :class:`collections.abc.Hashable` answers no rather than the call
    raising.

    **Not JSON.** ``json.dumps`` serializes a ``dict`` and nothing else that
    is a mapping. :func:`thaw_structure` gives one back, nested values
    included -- ``dict(m)`` thaws the top level only.

    **A subclass pickles as its own type but by the data alone**: unpickling
    and ``copy.deepcopy`` rebuild it from the frozen items without calling
    its ``__init__``, so state a subclass keeps beside those items is not
    carried.

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
        walk = _FreezeWalk()
        self._data = {key: walk.freeze(value, f"value[{key!r}]") for key, value in source.items()}

    @classmethod
    def _adopt(cls, data: dict[Any, Any]) -> FrozenMapping[Any, Any]:
        """Wrap ``data``, whose values are already frozen, without walking it again.

        Also what an unpickle calls, by this name -- see :meth:`__reduce__` --
        so renaming it breaks every pickle already written.
        """
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

    ======================== ==========================================
    Handed                   Returned
    ======================== ==========================================
    a ``Mapping``            a :class:`FrozenMapping` of frozen values
    a ``list`` or ``tuple``  a ``tuple`` of frozen values
    a ``set``                a ``frozenset``
    anything else            the same object
    ======================== ==========================================

    It reaches more than :func:`copy_structure`, which rebuilds dicts and
    lists only: any ``Mapping``, any ``list`` or ``tuple`` -- subclasses
    included, so a named tuple that needed rebuilding, or a ``list``
    subclass, comes back a plain ``tuple`` -- and a ``set``. A container of
    another kind, such as a ``deque`` or a ``bytearray``, is a leaf.

    **Leaves pass through, for** :func:`copy_structure`'s **reason**: a live
    object duplicated silently gives its owner a second one. So a mutable
    leaf stays mutable, and this is a freeze of the *structure* rather than
    of everything reachable.

    **An already-frozen value comes back as itself**: a ``FrozenMapping``
    (whose construction froze it), a ``frozenset``, and a tuple whose items
    are all already frozen. So freezing twice copies nothing.

    **A subtree shared between two places is frozen once**, and the frozen
    copy is shared the same way, so the walk is linear in the distinct
    containers it meets. Without that, a structure sharing a subtree at each
    level -- what YAML anchors produce -- would be expanded into real tuples
    once per path to each leaf.

    **A cycle is refused, not reproduced.** This is where the walk departs
    from :func:`copy_structure`, whose memo rebuilds a self-referring
    structure the way ``copy.deepcopy`` does. A mutable copy can be made to
    hold itself; a tuple or a ``FrozenMapping`` cannot, because each is
    complete before anything can hold it.

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
    return _FreezeWalk().freeze(value, "value")


class _FreezeWalk:
    """One :func:`freeze_structure` call's state.

    ``active`` holds the ids of the containers *on the current path*, which
    is what tells a cycle from a shared subtree: an id is removed on the way
    back out. ``done`` maps the id of each container already frozen to its
    frozen copy, so a second path to it reuses the copy. ``keep`` holds every
    container either one is keyed on, because an id is only unique among live
    objects: a ``Mapping`` computing its values could hand out a temporary
    whose id a later one reuses. The same reason ``copy_structure`` keeps its
    sources alive.
    """

    __slots__ = ("active", "done", "keep")

    def __init__(self) -> None:
        self.active: set[int] = set()
        self.done: dict[int, Any] = {}
        self.keep: list[Any] = []

    def freeze(self, value: Any, path: str) -> Any:
        if isinstance(value, (FrozenMapping, frozenset)):
            return value
        if not isinstance(value, (Mapping, list, tuple, set)):
            return value
        marker = id(value)
        if marker in self.active:
            raise ValueError(f"{path} holds itself, so it cannot be frozen.")
        if marker in self.done:
            return self.done[marker]
        self.keep.append(value)
        self.active.add(marker)
        try:
            frozen = self._rebuild(value, path)
        finally:
            self.active.discard(marker)
        self.done[marker] = frozen
        return frozen

    def _rebuild(self, value: Any, path: str) -> Any:
        if isinstance(value, Mapping):
            return FrozenMapping._adopt(
                {key: self.freeze(item, f"{path}[{key!r}]") for key, item in value.items()}
            )
        if isinstance(value, set):
            return frozenset(value)
        items = tuple(self.freeze(item, f"{path}[{index}]") for index, item in enumerate(value))
        if isinstance(value, tuple) and all(a is b for a, b in zip(items, value, strict=True)):
            return value
        return items


def thaw_structure(value: Any, seen: dict[int, Any] | None = None) -> Any:
    """Rebuild nested containers as plain dicts, lists and sets.

    The inverse of :func:`freeze_structure`, for the two things a frozen value
    cannot do: be serialized by ``json`` -- which takes a ``dict`` and a
    ``list`` and no other mapping or sequence -- and be changed. Every
    ``Mapping`` becomes a ``dict``, every ``list`` or ``tuple`` a ``list``,
    every ``set`` or ``frozenset`` a ``set``, and anything else passes
    through for :func:`copy_structure`'s reason.

    A tuple that was a tuple *before* it was frozen comes back a list too:
    once frozen, nothing records which tuples were lists, and a list is what
    serializes. A named tuple comes back a plain list.

    **The result is always new**, so changing it never reaches the value
    thawed. ``seen`` is :func:`copy_structure`'s memo, for its reasons: a
    container reached twice is thawed once and shared the same way, and a
    mutable value that holds itself terminates.

    Args:
        value: The value to thaw. Any type.
        seen: Optional memo mapping ``id()`` to the result already built
            for it, as :func:`copy_structure` takes.

    Returns:
        The value with its mappings, sequences and sets rebuilt as ``dict``,
        ``list`` and ``set``, and every other value the original.

    Example:
        ```python
        import json

        from dataknobs_common import freeze_structure, thaw_structure

        frozen = freeze_structure({"tags": ["hound"], "inner": {"k": 1}})
        assert thaw_structure(frozen) == {"tags": ["hound"], "inner": {"k": 1}}
        json.dumps(thaw_structure(frozen))   # json.dumps(frozen) would raise
        ```
    """
    if seen is None:
        seen = {}
    marker = id(value)
    if marker in seen:
        return seen[marker]
    if not isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return value
    # Hold the source so its id cannot be reused while the memo lives; see
    # ``copy_structure``.
    seen.setdefault(id(seen), []).append(value)
    if isinstance(value, Mapping):
        thawed_dict: dict[Any, Any] = {}
        seen[marker] = thawed_dict
        for key, item in value.items():
            thawed_dict[key] = thaw_structure(item, seen)
        return thawed_dict
    if isinstance(value, (set, frozenset)):
        thawed_set = set(value)
        seen[marker] = thawed_set
        return thawed_set
    thawed_list: list[Any] = []
    seen[marker] = thawed_list
    for item in value:
        thawed_list.append(thaw_structure(item, seen))
    return thawed_list

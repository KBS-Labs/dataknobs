# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Serialization protocols and utilities for dataknobs packages.

This module provides standard interfaces for objects that can be serialized
to and from dictionaries. This enables consistent serialization patterns
across all dataknobs packages.

The serialization framework supports:
- Type-safe protocols for serializable objects
- Utility functions for serialization/deserialization
- Runtime type checking with isinstance()
- Integration with dataclasses and custom classes

Example:
    ```python
    from dataknobs_common.serialization import Serializable
    from dataclasses import dataclass

    @dataclass
    class User:
        name: str
        email: str

        def to_dict(self) -> dict:
            return {"name": self.name, "email": self.email}

        @classmethod
        def from_dict(cls, data: dict) -> "User":
            return cls(name=data["name"], email=data["email"])

    # Type checking works
    user = User("Alice", "alice@example.com")
    assert isinstance(user, Serializable)  # True

    # Use utilities
    from dataknobs_common.serialization import serialize, deserialize

    data = serialize(user)
    restored = deserialize(User, data)
    ```
"""

import dataclasses
import datetime
import json
import logging
import math
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import (
    Any,
    ClassVar,
    Dict,
    Literal,
    NamedTuple,
    NoReturn,
    Protocol,
    Type,
    TypeVar,
    cast,
    runtime_checkable,
)

from dataknobs_common.exceptions import SerializationError

logger = logging.getLogger(__name__)

T = TypeVar("T")


@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be serialized to/from dict.

    Implement this protocol by providing to_dict() and from_dict() methods.
    The @runtime_checkable decorator allows isinstance() checks at runtime.

    Methods:
        to_dict: Convert object to dictionary representation
        from_dict: Create object from dictionary representation

    Example:
        ```python
        class MyClass:
            def __init__(self, value: str):
                self.value = value

            def to_dict(self) -> dict:
                return {"value": self.value}

            @classmethod
            def from_dict(cls, data: dict) -> "MyClass":
                return cls(data["value"])

        obj = MyClass("test")
        isinstance(obj, Serializable)
        # True
        ```
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary representation.

        Returns:
            Dictionary with serialized data

        Raises:
            SerializationError: If serialization fails
        """
        ...

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create object from dictionary representation.

        Args:
            data: Dictionary with serialized data

        Returns:
            Deserialized object instance

        Raises:
            SerializationError: If deserialization fails
        """
        ...


def serialize(obj: Any) -> Dict[str, Any]:
    """Serialize an object to dictionary.

    Convenience function that calls to_dict() with error handling.

    Args:
        obj: Object to serialize (must have to_dict method)

    Returns:
        Serialized dictionary

    Raises:
        SerializationError: If object doesn't support serialization or serialization fails

    Example:
        ```python
        class Point:
            def __init__(self, x: int, y: int):
                self.x, self.y = x, y
            def to_dict(self):
                return {"x": self.x, "y": self.y}

        point = Point(10, 20)
        data = serialize(point)
        # {'x': 10, 'y': 20}
        ```
    """
    if not hasattr(obj, "to_dict"):
        raise SerializationError(
            f"Object of type {type(obj).__name__} is not serializable (missing to_dict method)",
            context={"type": type(obj).__name__, "object": str(obj)},
        )

    try:
        result = obj.to_dict()
        if not isinstance(result, dict):
            raise SerializationError(
                f"to_dict() must return a dict, got {type(result).__name__}",
                context={"type": type(obj).__name__, "result_type": type(result).__name__},
            )
        return result
    except Exception as e:
        if isinstance(e, SerializationError):
            raise
        # Bounded message and context: `to_dict()` is the object's own code,
        # so its failure text is whatever that object's dependencies say. The
        # class name is bounded; the original travels on __cause__.
        raise SerializationError(
            f"Failed to serialize {type(obj).__name__} ({type(e).__name__})",
            context={"type": type(obj).__name__, "error_type": type(e).__name__},
        ) from e


def deserialize(cls: Type[T], data: Dict[str, Any]) -> T:
    """Deserialize dictionary into an object.

    Convenience function that calls from_dict() with error handling.

    Args:
        cls: Class to deserialize into (must have from_dict classmethod)
        data: Dictionary with serialized data

    Returns:
        Deserialized object instance

    Raises:
        SerializationError: If class doesn't support deserialization or deserialization fails

    Example:
        ```python
        class Point:
            def __init__(self, x: int, y: int):
                self.x, self.y = x, y
            @classmethod
            def from_dict(cls, data: dict):
                return cls(data["x"], data["y"])

        data = {"x": 10, "y": 20}
        point = deserialize(Point, data)
        # point.x, point.y
        # (10, 20)
        ```
    """
    if not hasattr(cls, "from_dict"):
        raise SerializationError(
            f"Class {cls.__name__} is not deserializable (missing from_dict classmethod)",
            context={"class": cls.__name__},
        )

    if not isinstance(data, dict):
        raise SerializationError(
            f"Data must be a dict, got {type(data).__name__}",
            context={"class": cls.__name__, "data_type": type(data).__name__},
        )

    try:
        # The hasattr guard above is what establishes ``from_dict``; this names
        # the shape it established. ``Type[T]`` alone does not carry it, so the
        # call was unchecked and its result leaked out as ``Any``.
        deserializable = cast("type[Serializable]", cls)
        return cast("T", deserializable.from_dict(data))
    except Exception as e:
        if isinstance(e, SerializationError):
            raise
        # Bounded message and context. `data` was the payload being decoded —
        # the caller already has it, and echoing it back into an error meant a
        # failed deserialize carried the whole document, whatever was in it.
        raise SerializationError(
            f"Failed to deserialize {cls.__name__} ({type(e).__name__})",
            context={"class": cls.__name__, "error_type": type(e).__name__},
        ) from e


def serialize_list(items: list[Any]) -> list[Dict[str, Any]]:
    """Serialize a list of objects to list of dictionaries.

    Args:
        items: List of serializable objects

    Returns:
        List of serialized dictionaries

    Raises:
        SerializationError: If any item cannot be serialized

    Example:
        ```python
        items = [Point(1, 2), Point(3, 4)]
        data_list = serialize_list(items)
        len(data_list)
        # 2
        ```
    """
    return [serialize(item) for item in items]


def deserialize_list(cls: Type[T], data_list: list[Dict[str, Any]]) -> list[T]:
    """Deserialize a list of dictionaries into objects.

    Args:
        cls: Class to deserialize into
        data_list: List of serialized dictionaries

    Returns:
        List of deserialized objects

    Raises:
        SerializationError: If any item cannot be deserialized

    Example:
        ```python
        data_list = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        points = deserialize_list(Point, data_list)
        len(points)
        # 2
        ```
    """
    return [deserialize(cls, data) for data in data_list]


def is_serializable(obj: Any) -> bool:
    """Check if an object is serializable.

    Args:
        obj: Object to check

    Returns:
        True if object has to_dict method

    Example:
        ```python
        class Point:
            def to_dict(self): return {}

        is_serializable(Point())
        # True
        is_serializable("string")
        # False
        ```
    """
    return isinstance(obj, Serializable) or hasattr(obj, "to_dict")


def is_deserializable(cls: Type) -> bool:
    """Check if a class is deserializable.

    Args:
        cls: Class to check

    Returns:
        True if class has from_dict classmethod

    Example:
        ```python
        class Point:
            @classmethod
            def from_dict(cls, data): return cls()

        is_deserializable(Point)
        # True
        is_deserializable(str)
        # False
        ```
    """
    return hasattr(cls, "from_dict")


#: How ``sanitize_for_json`` reports a value it drops.
OnDrop = Literal["silent", "warn", "error"]

#: A type, and how :func:`as_data` turns a value of it into data.
Using = Mapping[type, Callable[[Any], Any]]

_ON_DROP_MODES: tuple[str, ...] = ("silent", "warn", "error")

#: One step from a container to a value inside it: ``("name", field)``,
#: ``("key", key)``, ``("index", i)`` or ``("member", None)`` for a set's item.
_Step = tuple[str, Any]
_Path = tuple[_Step, ...]


class _Again(NamedTuple):
    """A rule's answer: walk *value* in place of the value it was given.

    An ``Enum``'s value, a ``to_dict()`` result, a serialiser's result. The
    value it came from stays open while *value* is walked, so a result that
    holds its own source is a cycle. *exempt* skips the ``using`` table for
    *value* alone, so a serialiser is not applied to its own result.
    """

    value: Any
    exempt: bool = False


class _Open(NamedTuple):
    """A rule's answer: a container, as its children and how to assemble them.

    *children* yields ``(step, key, child)``; each child is walked and, unless
    dropped, kept as ``(key, data)``. *finish* turns those pairs into the
    container's data. The children are yielded lazily, so a rule that reports
    something about an entry (a key it cannot write) does so in entry order.
    """

    children: Iterable[tuple[_Step, Any, Any]]
    finish: Callable[[list[tuple[Any, Any]]], Any]


def _as_dict(items: list[tuple[Any, Any]]) -> dict[Any, Any]:
    return dict(items)


def _as_list(items: list[tuple[Any, Any]]) -> list[Any]:
    return [item for _, item in items]


def _as_sorted_list(items: list[tuple[Any, Any]]) -> list[Any]:
    # Sorted by each item's JSON, so the same set always gives the same list.
    return sorted(_as_list(items), key=lambda item: json.dumps(item, sort_keys=True))


class _Walk:
    """One traversal of a value into JSON-shaped data, under one policy.

    A policy is an ordered tuple of rules. A rule only *matches*: it returns
    the value's data when that needs no walking (a JSON scalar), an
    :class:`_Again` naming a value to walk in its place, an :class:`_Open`
    describing a container, or ``_NO_RULE``. Every recursion happens in
    :meth:`walk`, so one level of nesting costs one stack frame and the walk
    follows nesting as deep as a plain recursive function would.

    What every policy shares lives here: the rules for an ``Enum``, a
    non-finite float, a JSON scalar, a list or tuple and a dataclass's
    fields; the path, kept as steps and written out only when something is
    reported; and the guard against a
    value that holds itself. The guard records the values open on the
    current path, not every value seen, so a value reached by two routes is
    walked both times and only a true cycle is reported.
    """

    #: The policy's rules, in order. Each takes ``(walk, value, path)``.
    _rules: ClassVar[tuple[Callable[..., Any], ...]] = ()

    def __init__(self) -> None:
        self._open: set[int] = set()
        self._exempt = False

    def walk(self, value: Any, path: _Path) -> Any:
        opened: list[int] = []
        exempt = False
        try:
            while True:
                self._exempt = exempt
                found = self._match(value, path)
                if isinstance(found, _Again):
                    if found.value is not value and not self._enter(value, opened):
                        return self._cycle(value, path)
                    value, exempt = found.value, found.exempt
                    continue
                if isinstance(found, _Open):
                    if not self._enter(value, opened):
                        return self._cycle(value, path)
                    items: list[tuple[Any, Any]] = []
                    for step, key, child in found.children:
                        item = self.walk(child, (*path, step))
                        if item is not _SENTINEL:
                            items.append((key, item))
                    return found.finish(items)
                return found
        finally:
            for marker in opened:
                self._open.discard(marker)

    def _match(self, value: Any, path: _Path) -> Any:
        for rule in self._rules:
            found = rule(self, value, path)
            if found is not _NO_RULE:
                return found
        return self._no_rule(value, path)

    def _enter(self, value: Any, opened: list[int]) -> bool:
        """Record *value* as open on the current path; ``False`` if it already is."""
        marker = id(value)
        if marker in self._open:
            return False
        self._open.add(marker)
        opened.append(marker)
        return True

    # -- policy hooks ------------------------------------------------------

    def _no_rule(self, value: Any, path: _Path) -> Any:
        raise NotImplementedError

    def _cycle(self, value: Any, path: _Path) -> Any:
        raise NotImplementedError

    def _non_finite(self, value: float, path: _Path) -> Any:
        raise NotImplementedError

    # -- shared rules ------------------------------------------------------

    def _enum(self, value: Any, path: _Path) -> Any:
        # Before the scalar rule, so an IntEnum / StrEnum (which subclass
        # int / str) becomes a plain int / str rather than the member.
        if isinstance(value, Enum):
            return _Again(value.value)
        return _NO_RULE

    def _finite(self, value: Any, path: _Path) -> Any:
        # Before the scalar rule: ``json.dumps`` writes these as ``NaN`` and
        # ``Infinity``, which are not JSON, and a strict reader refuses them.
        if isinstance(value, float) and not math.isfinite(value):
            return self._non_finite(value, path)
        return _NO_RULE

    def _scalar(self, value: Any, path: _Path) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return _NO_RULE

    def _sequence(self, value: Any, path: _Path) -> Any:
        if not isinstance(value, (list, tuple)):
            return _NO_RULE
        return _Open(((("index", i), None, item) for i, item in enumerate(value)), _as_list)

    def _set(self, value: Any, path: _Path) -> Any:
        # A list, sorted by each member's JSON, so the same set always gives
        # the same list.
        if not isinstance(value, (set, frozenset)):
            return _NO_RULE
        return _Open(((("member", None), None, item) for item in value), _as_sorted_list)

    def _fields(self, value: Any, path: _Path) -> Any:
        # Field by field rather than ``dataclasses.asdict()``, which deep-copies
        # every field value and so crashes on one that cannot be copied (an
        # ``asyncio.Task`` held in wizard state, say). Each field passes
        # through the whole walk instead.
        if not dataclasses.is_dataclass(value) or isinstance(value, type):
            return _NO_RULE
        children = (
            (("name", f.name), f.name, getattr(value, f.name)) for f in dataclasses.fields(value)
        )
        return _Open(children, _as_dict)


class _Sanitize(_Walk):
    """The policy of :func:`sanitize_for_json`: what has no JSON form is dropped.

    A dropped value is the private ``_SENTINEL``, which its container leaves
    out, and is reported according to *on_drop*; under ``"error"`` each drop
    is added to *dropped*. Paths are written ``a.b[0]``, a set member ``s{}``, with ``<root>`` for
    the value itself. *quiet* leaves out the DEBUG lines for a ``to_dict()``
    that raised and for a non-finite float, for a caller that only asks.
    """

    def __init__(self, on_drop: str, dropped: list[str], *, quiet: bool = False) -> None:
        super().__init__()
        self._on_drop = on_drop
        self._dropped = dropped
        self._quiet = quiet

    def _label(self, path: _Path) -> str:
        text = ""
        for kind, part in path:
            if kind == "index":
                text = f"{text}[{part}]"
            elif kind == "member":
                # A set has no order to give its member a position by.
                text = f"{text}{{}}"
            else:
                text = f"{text}.{part}" if text else str(part)
        # Only the value itself is labelled <root>. A key whose text is empty,
        # at the top, keeps its empty label, as it always has.
        return text if path else text or "<root>"

    def _drop(self, path: _Path, what: str) -> Any:
        label = self._label(path)
        if self._on_drop == "warn":
            logger.warning("Dropping non-serializable '%s' (%s)", label, what)
        elif self._on_drop == "error":
            self._dropped.append(f"{label} ({what})")
        else:
            logger.debug("Dropping non-serializable value at '%s' (%s)", label, what)
        return _SENTINEL

    def _no_rule(self, value: Any, path: _Path) -> Any:
        return self._drop(path, f"type={type(value).__name__}")

    def _cycle(self, value: Any, path: _Path) -> Any:
        return self._drop(path, "cycle")

    def _non_finite(self, value: float, path: _Path) -> Any:
        # Written as null, as ``JSON.stringify`` writes it, rather than
        # dropped: a drop takes the rest of a record's key with it in a
        # caller that keeps only what nothing was dropped from, and moves
        # every later item of a list down one place. Converted, so not
        # reported, as an Enum is not.
        if not self._quiet:
            logger.debug(
                "Writing non-finite float %r at '%s' as null", float(value), self._label(path)
            )
        return None

    def _dict(self, value: Any, path: _Path) -> Any:
        if not isinstance(value, dict):
            return _NO_RULE

        def children() -> Iterator[tuple[_Step, Any, Any]]:
            for k, v in value.items():
                key = _json_key(k)
                if key is _SENTINEL:
                    self._drop((*path, ("key", k)), _unwritable_key(k))
                    continue
                yield ("key", key), key, v

        return _Open(children(), _as_dict)

    def _to_dict(self, value: Any, path: _Path) -> Any:
        if not (hasattr(value, "to_dict") and callable(value.to_dict)):
            return _NO_RULE
        try:
            result = value.to_dict()
        except Exception:
            if not self._quiet:
                logger.debug("to_dict() failed for %s, dropping", type(value).__name__)
            return _NO_RULE
        if not isinstance(result, dict):
            return _NO_RULE
        return _Again(result)

    def _text_form(self, value: Any, path: _Path) -> Any:
        # Values with one standard text form, written in it, as Django's and
        # pydantic's JSON encoders write them: a date or time in ISO 8601, a
        # Decimal as its digits (a float would round it), a UUID and a path
        # as their string. A conversion, so not reported; it is lossy in
        # type, so each reads back as a ``str``. A NaN or infinite Decimal is
        # written as null, as a float one is.
        if isinstance(value, (datetime.date, datetime.time)):
            return value.isoformat()
        if isinstance(value, Decimal):
            if not value.is_finite():
                # float() raises on a signalling NaN, so a NaN is passed as one.
                return self._non_finite(float(value) if value.is_infinite() else math.nan, path)
            return str(value)
        if isinstance(value, (uuid.UUID, PurePath)):
            return str(value)
        return _NO_RULE

    _rules = (
        _Walk._enum,
        _Walk._finite,
        _Walk._scalar,
        _dict,
        _Walk._sequence,
        _Walk._set,
        _Walk._fields,
        _text_form,
        _to_dict,
    )


class _AsData(_Walk):
    """The policy of :func:`as_data`: what has no rule is refused, at once.

    Paths are written ``value.a['b'][0]``: an attribute after a dot, a key
    as written in brackets, so the two cannot be confused. A set's member is
    written ``{}``, since a set has no order to give it a position by.
    """

    def __init__(self, using: Using | None, root: str) -> None:
        super().__init__()
        self._using: Using = using or {}
        self._root = root
        for kind in self._using:
            if not isinstance(kind, type):
                raise SerializationError(
                    f"using takes types as keys, not {kind!r}",
                    context={"key_type": type(kind).__name__},
                )

    def walk_fields(self, value: Any, names: list[str], replaced: Mapping[str, Any]) -> Any:
        """*value*'s fields, each named in *replaced* given its replacement."""
        opened: list[int] = []
        self._enter(value, opened)
        try:
            return {
                name: self.walk(
                    replaced[name] if name in replaced else getattr(value, name),
                    (("name", name),),
                )
                for name in names
            }
        finally:
            self._open.discard(opened[0])

    def _label(self, path: _Path) -> str:
        text = self._root
        for kind, part in path:
            if kind == "name":
                text = f"{text}.{part}"
            elif kind == "key":
                text = f"{text}[{part!r}]"
            elif kind == "index":
                text = f"{text}[{part}]"
            else:
                text = f"{text}{{}}"
        return text

    def _refuse(self, path: _Path, message: str, **context: Any) -> NoReturn:
        where = self._label(path)
        raise SerializationError(f"{where}{message}", context={"path": where, **context})

    def _refuse_raised(self, path: _Path, what: str, error: Exception) -> NoReturn:
        # Bounded: the message names the exception's class, not its text,
        # which is whatever the caller's own code put there. The original,
        # with its own path when it is a refusal from a nested walk, travels
        # on __cause__.
        where = self._label(path)
        raise SerializationError(
            f"{where}: {what} raised {type(error).__name__}",
            context={"path": where, "error_type": type(error).__name__},
        ) from error

    def _no_rule(self, value: Any, path: _Path) -> Any:
        kind = type(value).__name__
        self._refuse(path, f" is of type {kind}, which has no rule as data", type=kind)

    def _cycle(self, value: Any, path: _Path) -> Any:
        self._refuse(path, " refers back to a value that contains it")

    def _from_using(self, value: Any, path: _Path) -> Any:
        if self._exempt:
            return _NO_RULE
        found = self._lookup(type(value))
        if found is None:
            return _NO_RULE
        kind, convert = found
        try:
            result = convert(value)
        except Exception as e:
            self._refuse_raised(path, f"the serialiser for {kind.__name__}", e)
        # The serialiser is not applied to the value it returned, so one that
        # returns its own type cannot loop. A value of that type anywhere
        # inside the result is still shown by it.
        return _Again(result, exempt=True)

    def _lookup(self, kind: type) -> tuple[type, Callable[[Any], Any]] | None:
        # The exact type first, so a subclass listed after its base is reached;
        # then the first base listed.
        if kind in self._using:
            return kind, self._using[kind]
        for base, convert in self._using.items():
            if issubclass(kind, base):
                return base, convert
        return None

    def _non_finite(self, value: float, path: _Path) -> Any:
        self._refuse(
            path,
            f" is a non-finite float ({float(value)!r}), which JSON cannot write",
            type="float",
        )

    def _to_dict(self, value: Any, path: _Path) -> Any:
        # Before a dataclass's fields: a value that chose its own shape keeps it.
        to_dict = getattr(value, "to_dict", None)
        if not callable(to_dict) or isinstance(value, type):
            return _NO_RULE
        try:
            result = to_dict()
        except Exception as e:
            self._refuse_raised(path, f"to_dict() of {type(value).__name__}", e)
        if result is value:
            self._refuse(path, f": to_dict() of {type(value).__name__} returned the value itself")
        return _Again(result)

    def _mapping(self, value: Any, path: _Path) -> Any:
        if not isinstance(value, Mapping):
            return _NO_RULE

        def children() -> Iterator[tuple[_Step, Any, Any]]:
            written: dict[str, Any] = {}
            for key, item in value.items():
                name = _text_key(key)
                if name in written:
                    self._refuse(
                        path,
                        f" has two keys written {name!r}: {written[name]!r} and {key!r}",
                        key=name,
                    )
                written[name] = key
                yield ("key", key), name, item

        return _Open(children(), _as_dict)

    _rules = (
        _from_using,
        _Walk._enum,
        _Walk._finite,
        _Walk._scalar,
        _to_dict,
        _Walk._fields,
        _mapping,
        _Walk._sequence,
        _Walk._set,
    )


def sanitize_for_json(value: Any, on_drop: OnDrop = "silent") -> Any:
    """Recursively ensure a value is JSON-serializable.

    Converts known structured types (dataclasses, Serializable objects) to
    dicts and drops anything that cannot be represented in JSON. Designed
    for wizard state data that may contain a mix of serializable primitives,
    dataclass instances, Serializable objects, and live runtime objects.
    For data that must not lose anything, use :func:`as_data`, which refuses
    where this drops.

    Conversion rules (in order):
    - ``Enum`` member → its ``.value`` (recursively sanitized), so
      ``StrEnum`` / ``IntEnum`` collapse to a plain ``str`` / ``int``
      rather than leaking an enum instance into the output
    - a NaN or infinite ``float`` → ``None``, as ``JSON.stringify`` writes
      it, since JSON has no form for it (``json.dumps`` would write ``NaN``
      / ``Infinity``, which a strict reader refuses). This is a conversion,
      not a drop: it is not reported, and a list keeps every position. It
      is lossy, so NaN, an infinity and ``None`` read back the same
    - ``None``, ``bool``, ``int``, ``float``, ``str`` → pass through
    - ``dict`` → recurse on values; drop entries whose values are not
      convertible. A key is kept if it is a ``str``, ``int``, finite
      ``float``, ``bool`` or ``None`` (the keys ``json.dumps`` accepts); an
      ``Enum`` key becomes its value; an entry with any other key, a NaN or
      infinite float among them, is dropped
    - ``list`` / ``tuple`` → recurse on elements; filter out non-convertible items
    - ``set`` / ``frozenset`` → a list of its members, recursively, sorted by
      each member's JSON so the same set always gives the same list
    - dataclass instance → field-by-field recursive sanitization. This rule
      comes first, so a dataclass's own ``to_dict()`` is not called
    - ``datetime``, ``date``, ``time`` → ``isoformat()``; ``Decimal`` →
      ``str()`` (a NaN or infinite one → ``None``); ``UUID`` and a path →
      ``str()``. Conversions, not drops, and lossy in type: each reads back
      as a ``str``
    - Object with ``to_dict()`` returning a ``dict`` → that dict, recursively
    - Anything else → dropped (behaviour controlled by *on_drop*)

    A dropped value is left out of the ``dict``, list or dataclass that
    holds it. A value dropped at the top level has no container to be
    left out of, so the result is ``None`` (or, with *on_drop* ``"error"``,
    :class:`SerializationError` is raised). A container that holds itself
    is dropped where it repeats, reported as a ``cycle``; a value reached
    twice by different routes is not a cycle and is kept both times.

    Args:
        value: Any Python value to sanitize.
        on_drop: What to do when a non-serializable value is encountered.
            ``"silent"`` (default) — log at DEBUG (backward-compatible).
            ``"warn"`` — log at WARNING with the key path.
            ``"error"`` — collect all dropped paths and raise
            :class:`SerializationError` after traversal.

    Returns:
        A JSON-safe version of *value*; ``None`` when *value* itself is
        dropped.

    Raises:
        SerializationError: When *on_drop* is ``"error"`` and at least one
            non-serializable value is encountered.
        ValueError: When *on_drop* is not one of the three modes.
    """
    if on_drop not in _ON_DROP_MODES:
        raise ValueError(f"on_drop must be one of 'silent', 'warn', 'error'; got {on_drop!r}")

    dropped: list[str] = []
    result = _Sanitize(on_drop, dropped).walk(value, ())

    if dropped:
        raise SerializationError(
            f"Non-serializable values at: {', '.join(dropped)}",
            context={"dropped_paths": dropped},
        )

    # The walk marks a dropped value with a private sentinel, which its
    # containers filter out. A dropped top-level value has no container,
    # so the sentinel must not escape to the caller: it is not JSON.
    if result is _SENTINEL:
        return None
    return result


def _json_key(key: Any) -> Any:
    """Return *key* as a key ``json.dumps`` accepts, or ``_SENTINEL``.

    ``json.dumps`` writes ``str``, ``int``, ``float``, ``bool`` and ``None``
    keys and raises ``TypeError`` on any other. An ``Enum`` key is read as
    its value, as an ``Enum`` value is. A NaN or infinite float key is
    refused too: ``json.dumps`` writes it as ``"NaN"`` only when it may
    write non-finite floats at all, and two NaN keys are two entries that
    would be written as one.
    """
    while isinstance(key, Enum):
        key = key.value
    if isinstance(key, float) and not math.isfinite(key):
        return _SENTINEL
    if key is None or isinstance(key, (str, int, float, bool)):
        return key
    return _SENTINEL


def _unwritable_key(key: Any) -> str:
    """What is reported for a dict key :func:`_json_key` refused."""
    value = key
    while isinstance(value, Enum):
        value = value.value
    if isinstance(value, float):
        return f"non-finite float key {float(value)!r}"
    return f"key type={type(key).__name__}"


def _text_key(key: Any) -> str:
    """*key* as :func:`as_data` writes it.

    An ``Enum`` is read as its value first. A JSON scalar is written as
    ``json.dumps`` writes it as a key (``None`` as ``"null"``, ``True`` as
    ``"true"``, a float by its ``repr``), so the two agree on the same dict;
    any other key is written with ``str()``.
    """
    while isinstance(key, Enum):
        key = key.value
    if isinstance(key, str):
        return key
    if key is None or isinstance(key, (bool, int, float)):
        return str(next(iter(json.loads(json.dumps({key: None})))))
    return str(key)


def sanitize_and_report(value: Any) -> tuple[Any, list[str]]:
    """What :func:`sanitize_for_json` returns, and what it would report, in one walk.

    For a caller that decides by the report what to do with the result —
    keeping a value only when nothing was dropped from it, say — and would
    otherwise walk it twice, calling each ``to_dict()`` twice. The rules are
    those of :func:`sanitize_for_json`; nothing is logged or raised.

    Args:
        value: Any Python value. It is never modified.

    Returns:
        ``(data, dropped)``: *value* as :func:`sanitize_for_json` returns it,
        and each dropped value's path and what was found there, as
        :func:`validate_json_safe` returns them. An empty *dropped* means
        *data* is all of *value*.
    """
    dropped: list[str] = []
    result = _Sanitize("error", dropped, quiet=True).walk(value, ())
    return (None if result is _SENTINEL else result), dropped


def validate_json_safe(value: Any) -> list[str]:
    """Return paths of non-serializable values without modifying the input.

    The same walk as :func:`sanitize_for_json` with ``on_drop="error"``,
    returning what that would raise for rather than raising: each entry is
    a path and what was found there, as in ``"a.b (type=function)"``. The
    input is never modified. A value the walk converts (an ``Enum``, a NaN)
    is not reported. To have the converted value as well, use
    :func:`sanitize_and_report`, which walks once for both.

    Args:
        value: Any Python value to check.

    Returns:
        List of key paths to non-serializable values.  An empty list
        means *value* is fully JSON-safe.
    """
    return sanitize_and_report(value)[1]


def as_data(value: Any, *, using: Using | None = None) -> Any:
    """*value* as JSON-shaped data, refused where it cannot be, never dropped.

    For a value that must arrive whole: a reply made of nested frozen
    dataclasses, wanted as JSON for a report, a test or another process.
    Where :func:`sanitize_for_json` drops what it cannot convert, this
    raises, naming where the value was, because data that lost a field
    without saying so reads as data where that field was empty.

    Rules (in order):

    1. a type in *using* → that type's serialiser, its result walked. The
       serialiser is not applied to the value it returned, so one returning
       its own type cannot loop, but a value of that type anywhere inside
       the result is shown by it too. The exact type is looked up first,
       then the first listed base;
    2. an ``Enum`` → its value, walked;
    3. ``None``, ``bool``, ``int``, ``float``, ``str`` → itself, except a
       NaN or infinite float, which JSON cannot write and is refused;
    4. an object with a callable ``to_dict()`` (not a class) → its result,
       whatever it is, walked. This comes before rule 5, so a dataclass that
       chose its own shape keeps it;
    5. a dataclass instance → its fields, in declaration order;
    6. any :class:`~collections.abc.Mapping` → its items. An ``Enum`` key
       is read as its value first; a JSON scalar key is written as
       ``json.dumps`` writes it (``None`` as ``"null"``, ``True`` as
       ``"true"``), and any other key with ``str()``. Two keys written the
       same, such as ``1`` and ``"1"``, are refused, because one would
       replace the other;
    7. a ``list`` or ``tuple`` → a list;
    8. a ``set`` or ``frozenset`` → a list, sorted by each item's JSON, so
       the same set always gives the same data;
    9. anything else → refused.

    Args:
        value: Any Python value.
        using: A per-call table of serialisers by type, for a value shown
            differently in one place (a supplier as its id, say) without
            giving its type a ``to_dict()`` every other caller would get.

    Returns:
        *value* as dicts with ``str`` keys, lists and JSON scalars.

    Raises:
        SerializationError: For a value no rule takes, two keys written the
            same, a non-finite float, a container that holds itself, a
            value nested past the interpreter's recursion limit, or a
            ``to_dict()`` or serialiser that raised (the original, a nested
            refusal included, is the ``__cause__``). The message names the
            path, as in ``value.notes['tare']``, and so does
            ``context["path"]``.
    """
    walk = _AsData(using, "value")
    try:
        return walk.walk(value, ())
    except RecursionError as e:
        raise _too_deep("value") from e


def fields_data(value: Any, /, *, using: Using | None = None, **replaced: Any) -> dict[str, Any]:
    """A dataclass's fields as data, those named in *replaced* given as passed.

    For a value with one field too large to carry whole:
    ``fields_data(order, lines={"count": 200})``. Each field, replaced or
    not, is walked by the rules of :func:`as_data`, and a refusal names the
    path from the class, as in ``Order.lines[0]``.

    Args:
        value: A dataclass instance.
        using: As for :func:`as_data`.
        **replaced: Field names, each with the value to give in its place.

    Returns:
        The fields, in declaration order.

    Raises:
        SerializationError: When *value* is not a dataclass instance; when a
            name in *replaced* is not one of its fields, so a renamed field
            cannot leave a replacement behind unnoticed; or as for
            :func:`as_data`.
    """
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        kind = type(value).__name__
        raise SerializationError(
            f"fields_data takes a dataclass instance, not {kind}", context={"type": kind}
        )
    kind = type(value).__name__
    names = [f.name for f in dataclasses.fields(value)]
    unknown = sorted(set(replaced) - set(names))
    if unknown:
        raise SerializationError(
            f"{kind} has no field {', '.join(unknown)}",
            context={"type": kind, "unknown_fields": unknown},
        )
    walk = _AsData(using, kind)
    try:
        result: dict[str, Any] = walk.walk_fields(value, names, replaced)
    except RecursionError as e:
        raise _too_deep(kind) from e
    return result


def _too_deep(root: str) -> SerializationError:
    """The refusal for a value nested past what the walk can follow."""
    return SerializationError(
        f"{root} is nested too deeply to walk (past the interpreter's recursion limit)",
        context={"path": root},
    )


def jsonify(value: Any) -> Any:
    """Recursively normalise ``Enum`` members to their ``.value``.

    Walks ``dict`` / ``list`` / ``tuple`` containers and replaces every
    ``Enum`` member with its ``.value``, recursing into that value as well
    (so an enum whose ``.value`` is itself an enum — or a container of
    enums — is fully normalised); all other values pass through unchanged.
    A *lossless* normaliser for data that is already JSON-shaped
    except for enums — typically the output of ``dataclasses.asdict`` or
    :meth:`~dataknobs_common.structured_config.StructuredConfig.to_dict`.
    ``StrEnum`` / ``IntEnum`` members collapse to plain ``str`` / ``int`` so
    ``json.dumps`` needs no custom encoder.

    Unlike :func:`sanitize_for_json`, ``jsonify`` does **not** drop
    non-serializable values, convert dataclasses / ``to_dict`` objects, or
    descend into arbitrary objects: callables, ``type`` objects, ``set``s,
    etc. pass through untouched (a structure still holding them is no more
    JSON-serialisable afterwards than before). Reach for
    :func:`sanitize_for_json` when you want lossy best-effort JSON safety
    with dataclass / ``Serializable`` conversion and drop semantics; reach
    for ``jsonify`` when you have a known-shaped structure whose only
    non-JSON-native values are enums and you need an exact round-trip.

    Args:
        value: Any value; commonly a ``dict`` / ``list`` tree from
            ``dataclasses.asdict``.

    Returns:
        The same structure with every ``Enum`` member replaced by its
        ``.value``.
    """
    if isinstance(value, Enum):
        # Recurse into the member's value so an enum whose ``.value`` is
        # itself an ``Enum`` (or a container of enums) is fully normalised,
        # as ``sanitize_for_json`` walks an enum's value.
        return jsonify(value.value)
    if isinstance(value, dict):
        return {k: jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonify(v) for v in value]
    if isinstance(value, tuple):
        return tuple(jsonify(v) for v in value)
    return value


# Internal sentinel to distinguish "value was dropped" from "value is None"
_SENTINEL = object()

# Internal marker: a rule of the walk does not apply to this value
_NO_RULE = object()


__all__ = [
    "Serializable",
    "sanitize_for_json",
    "sanitize_and_report",
    "validate_json_safe",
    "as_data",
    "fields_data",
    "Using",
    "OnDrop",
    "jsonify",
    "serialize",
    "deserialize",
    "serialize_list",
    "deserialize_list",
    "is_serializable",
    "is_deserializable",
]

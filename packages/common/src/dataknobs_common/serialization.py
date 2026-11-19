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
import json
import logging
from collections.abc import Callable, Mapping
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Dict,
    Literal,
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


class _Walk:
    """One traversal of a value into JSON-shaped data, under one policy.

    The walk tries its policy's rules in order and takes the first that
    applies; a value no rule takes goes to :meth:`_no_rule`. What every
    policy shares lives here: the rules for an ``Enum``, a JSON primitive, a
    list or tuple and a dataclass's fields; the path, kept as steps and
    written out only when something is reported; and the guard against a
    value that holds itself. A policy decides only what differs: which rules
    it has and in what order, how it reads a mapping and a ``to_dict()``,
    and what happens to a value it has no rule for.

    The guard records the containers on the current path, not every
    container seen, so a value reached by two routes is walked both times
    and only a true cycle is reported.
    """

    #: The policy's rules, in order. Each takes ``(walk, value, path)`` and
    #: returns the value as data, or ``_NO_RULE`` when it does not apply.
    _rules: ClassVar[tuple[Callable[..., Any], ...]] = ()

    def __init__(self) -> None:
        self._open: set[int] = set()

    def walk(self, value: Any, path: _Path) -> Any:
        for rule in self._rules:
            result = rule(self, value, path)
            if result is not _NO_RULE:
                return result
        return self._no_rule(value, path)

    # -- policy hooks ------------------------------------------------------

    def _no_rule(self, value: Any, path: _Path) -> Any:
        raise NotImplementedError

    def _cycle(self, value: Any, path: _Path) -> Any:
        raise NotImplementedError

    # -- shared machinery --------------------------------------------------

    def _inside(self, value: Any, path: _Path, build: Callable[[], Any]) -> Any:
        """``build()``, with *value* recorded as open on the current path."""
        marker = id(value)
        if marker in self._open:
            return self._cycle(value, path)
        self._open.add(marker)
        try:
            return build()
        finally:
            self._open.discard(marker)

    # -- shared rules ------------------------------------------------------

    def _enum(self, value: Any, path: _Path) -> Any:
        # Before the primitive rule, so an IntEnum / StrEnum (which subclass
        # int / str) becomes a plain int / str rather than the member.
        if isinstance(value, Enum):
            return self.walk(value.value, path)
        return _NO_RULE

    def _primitive(self, value: Any, path: _Path) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return _NO_RULE

    def _sequence(self, value: Any, path: _Path) -> Any:
        if not isinstance(value, (list, tuple)):
            return _NO_RULE

        def build() -> list[Any]:
            items = (self.walk(item, (*path, ("index", i))) for i, item in enumerate(value))
            return [item for item in items if item is not _SENTINEL]

        return self._inside(value, path, build)

    def _fields(self, value: Any, path: _Path) -> Any:
        # Field by field rather than ``dataclasses.asdict()``, which deep-copies
        # every field value and so crashes on one that cannot be copied (an
        # ``asyncio.Task`` held in wizard state, say). Each field passes
        # through the whole walk instead.
        if not dataclasses.is_dataclass(value) or isinstance(value, type):
            return _NO_RULE

        def build() -> dict[str, Any]:
            result: dict[str, Any] = {}
            for f in dataclasses.fields(value):
                item = self.walk(getattr(value, f.name), (*path, ("name", f.name)))
                if item is not _SENTINEL:
                    result[f.name] = item
            return result

        return self._inside(value, path, build)


class _Sanitize(_Walk):
    """The policy of :func:`sanitize_for_json`: what has no JSON form is dropped.

    A dropped value is the private ``_SENTINEL``, which its container leaves
    out, and is reported according to *on_drop*. Paths are written
    ``a.b[0]``, with ``<root>`` for the value itself.
    """

    def __init__(self, on_drop: str, base: str, dropped: list[str] | None) -> None:
        super().__init__()
        self._on_drop = on_drop
        self._base = base
        self._dropped = dropped

    def _label(self, path: _Path) -> str:
        text = self._base
        for kind, part in path:
            if kind == "index":
                text = f"{text}[{part}]"
            else:
                text = f"{text}.{part}" if text else str(part)
        return text or "<root>"

    def _drop(self, path: _Path, what: str) -> Any:
        label = self._label(path)
        if self._on_drop == "warn":
            logger.warning("Dropping non-serializable '%s' (%s)", label, what)
        elif self._on_drop == "error" and self._dropped is not None:
            self._dropped.append(f"{label} ({what})")
        else:
            logger.debug("Dropping non-serializable value at '%s' (%s)", label, what)
        return _SENTINEL

    def _no_rule(self, value: Any, path: _Path) -> Any:
        return self._drop(path, f"type={type(value).__name__}")

    def _cycle(self, value: Any, path: _Path) -> Any:
        return self._drop(path, "cycle")

    def _dict(self, value: Any, path: _Path) -> Any:
        if not isinstance(value, dict):
            return _NO_RULE

        def build() -> dict[Any, Any]:
            result: dict[Any, Any] = {}
            for k, v in value.items():
                key = _json_key(k)
                if key is _SENTINEL:
                    self._drop((*path, ("key", k)), f"key type={type(k).__name__}")
                    continue
                item = self.walk(v, (*path, ("key", key)))
                if item is not _SENTINEL:
                    result[key] = item
            return result

        return self._inside(value, path, build)

    def _to_dict(self, value: Any, path: _Path) -> Any:
        if not (hasattr(value, "to_dict") and callable(value.to_dict)):
            return _NO_RULE
        try:
            result = value.to_dict()
        except Exception:
            logger.debug("to_dict() failed for %s, dropping", type(value).__name__)
            return _NO_RULE
        if not isinstance(result, dict):
            return _NO_RULE
        return self._inside(value, path, lambda: self.walk(result, path))

    _rules = (
        _Walk._enum,
        _Walk._primitive,
        _dict,
        _Walk._sequence,
        _Walk._fields,
        _to_dict,
    )


class _AsData(_Walk):
    """The policy of :func:`as_data`: what has no rule is refused, at once.

    Paths are written ``value.a['b'][0]``: an attribute after a dot, a key
    as written in brackets, so the two cannot be confused.
    """

    def __init__(self, using: Using | None, root: str) -> None:
        super().__init__()
        self._using: Using = using or {}
        self._root = root

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

    def _no_rule(self, value: Any, path: _Path) -> Any:
        kind = type(value).__name__
        self._refuse(path, f" is of type {kind}, which has no rule as data", type=kind)

    def _cycle(self, value: Any, path: _Path) -> Any:
        self._refuse(path, " refers back to a value that contains it")

    def _from_using(self, value: Any, path: _Path) -> Any:
        found = self._lookup(type(value))
        if found is None:
            return _NO_RULE
        kind, convert = found
        try:
            result = convert(value)
        except SerializationError:
            raise
        except Exception as e:
            self._refuse_raised(path, f"the serialiser for {kind.__name__}", e)
        # Applied once: the type leaves the table for its own result, so a
        # serialiser returning a value of its own type cannot loop.
        table = self._using
        self._using = {k: c for k, c in table.items() if k is not kind}
        try:
            return self.walk(result, path)
        finally:
            self._using = table

    def _lookup(self, kind: type) -> tuple[type, Callable[[Any], Any]] | None:
        # The exact type first, so a subclass listed after its base is reached;
        # then the first base listed.
        if kind in self._using:
            return kind, self._using[kind]
        for base, convert in self._using.items():
            if issubclass(kind, base):
                return base, convert
        return None

    def _refuse_raised(self, path: _Path, what: str, error: Exception) -> NoReturn:
        # Bounded: the message names the exception's class, not its text,
        # which is whatever the caller's own code put there. The original
        # travels on __cause__.
        where = self._label(path)
        raise SerializationError(
            f"{where}: {what} raised {type(error).__name__}",
            context={"path": where, "error_type": type(error).__name__},
        ) from error

    def _to_dict(self, value: Any, path: _Path) -> Any:
        # Before a dataclass's fields: a value that chose its own shape keeps it.
        to_dict = getattr(value, "to_dict", None)
        if not callable(to_dict) or isinstance(value, type):
            return _NO_RULE
        try:
            result = to_dict()
        except SerializationError:
            raise
        except Exception as e:
            self._refuse_raised(path, f"to_dict() of {type(value).__name__}", e)
        return self._inside(value, path, lambda: self.walk(result, path))

    def _mapping(self, value: Any, path: _Path) -> Any:
        if not isinstance(value, Mapping):
            return _NO_RULE

        def build() -> dict[str, Any]:
            written: dict[str, Any] = {}
            result: dict[str, Any] = {}
            for key, item in value.items():
                name = _text_key(key)
                if name in written:
                    self._refuse(
                        path,
                        f" has two keys written {name!r}: {written[name]!r} and {key!r}",
                        key=name,
                    )
                written[name] = key
                result[name] = self.walk(item, (*path, ("key", key)))
            return result

        return self._inside(value, path, build)

    def _set(self, value: Any, path: _Path) -> Any:
        if not isinstance(value, (set, frozenset)):
            return _NO_RULE
        items = [self.walk(item, (*path, ("member", None))) for item in value]
        # Sorted by each item's JSON, so the same set always gives the same list.
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))

    _rules = (
        _from_using,
        _Walk._enum,
        _Walk._primitive,
        _to_dict,
        _Walk._fields,
        _mapping,
        _Walk._sequence,
        _set,
    )


def sanitize_for_json(
    value: Any,
    on_drop: OnDrop = "silent",
    _path: str = "",
    _dropped: list[str] | None = None,
) -> Any:
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
    - ``None``, ``bool``, ``int``, ``float``, ``str`` → pass through
    - ``dict`` → recurse on values; drop entries whose values are not
      convertible. A key is kept if it is a ``str``, ``int``, ``float``,
      ``bool`` or ``None`` (the keys ``json.dumps`` accepts); an ``Enum``
      key becomes its value; an entry with any other key is dropped
    - ``list`` / ``tuple`` → recurse on elements; filter out non-convertible items
    - dataclass instance → field-by-field recursive sanitization. This rule
      comes first, so a dataclass's own ``to_dict()`` is not called
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
        _path: Internal — dot-delimited key path for diagnostics.
        _dropped: Internal — accumulator for ``on_drop="error"`` mode.

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

    # For "error" mode the top-level caller owns the accumulator.
    is_top_level = _dropped is None and on_drop == "error"
    if is_top_level:
        _dropped = []

    result = _Sanitize(on_drop, _path, _dropped).walk(value, ())

    if is_top_level and _dropped:
        raise SerializationError(
            f"Non-serializable values at: {', '.join(_dropped)}",
            context={"dropped_paths": _dropped},
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
    its value, as an ``Enum`` value is.
    """
    while isinstance(key, Enum):
        key = key.value
    if key is None or isinstance(key, (str, int, float, bool)):
        return key
    return _SENTINEL


def _text_key(key: Any) -> str:
    """*key* as :func:`as_data` writes it: an ``Enum`` as its value, then ``str()``."""
    while isinstance(key, Enum):
        key = key.value
    return str(key)


def validate_json_safe(value: Any, _path: str = "") -> list[str]:
    """Return paths of non-serializable values without modifying the input.

    The same walk as :func:`sanitize_for_json` with ``on_drop="error"``,
    returning what that would raise for rather than raising: each entry is
    a path and what was found there, as in ``"a.b (type=function)"``. The
    input is never modified.

    Args:
        value: Any Python value to check.
        _path: Internal — dot-delimited key path for diagnostics.

    Returns:
        List of key paths to non-serializable values.  An empty list
        means *value* is fully JSON-safe.
    """
    paths: list[str] = []
    _Sanitize("error", _path, paths).walk(value, ())
    return paths


def as_data(value: Any, *, using: Using | None = None) -> Any:
    """*value* as JSON-shaped data, refused where it cannot be, never dropped.

    For a value that must arrive whole: a reply made of nested frozen
    dataclasses, wanted as JSON for a report, a test or another process.
    Where :func:`sanitize_for_json` drops what it cannot convert, this
    raises, naming where the value was, because data that lost a field
    without saying so reads as data where that field was empty.

    Rules (in order):

    1. a type in *using* → that type's serialiser, applied once: the type
       leaves the table for its result, so a serialiser that returns its own
       type cannot loop. The exact type is looked up first, then the first
       listed base;
    2. an ``Enum`` → its value, walked;
    3. ``None``, ``bool``, ``int``, ``float``, ``str`` → itself;
    4. an object with a callable ``to_dict()`` (not a class) → its result,
       whatever it is, walked. This comes before rule 5, so a dataclass that
       chose its own shape keeps it;
    5. a dataclass instance → its fields, in declaration order;
    6. any :class:`~collections.abc.Mapping` → its items, each key written
       as ``str()`` (an ``Enum`` key as its value first). Two keys written
       the same, such as ``1`` and ``"1"``, are refused, because one would
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
            same, a container that holds itself, or a ``to_dict()`` or
            serialiser that raised (the original is the ``__cause__``). The
            message names the path, as in ``value.notes['tare']``, and so
            does ``context["path"]``.
    """
    return _AsData(using, "value").walk(value, ())


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

    def build() -> dict[str, Any]:
        return {
            name: walk.walk(
                replaced[name] if name in replaced else getattr(value, name),
                (("name", name),),
            )
            for name in names
        }

    result: dict[str, Any] = walk._inside(value, (), build)
    return result


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

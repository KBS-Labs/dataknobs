# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""JSON Schemas for a model's reply, built closed.

A provider that can constrain its output to a schema does so when one is set
on ``LLMConfig.response_schema`` (``dataknobs_llm``). The providers differ on
which schemas they accept:

* **OpenAI** enforces a schema only in strict mode
  (``LLMConfig.response_schema_strict``, on by default), and strict mode takes
  only objects that list every property in ``required`` and set
  ``additionalProperties`` to false.
* **Anthropic** always enforces a schema, sent as ``output_config.format``,
  and requires every object to set ``additionalProperties`` to false.
* **Ollama** sends it as ``format`` and accepts any schema.

So a schema that leaves an object open works under Ollama and is refused with
a 400 by OpenAI and Anthropic, and one that leaves a property optional is
refused by OpenAI's strict mode: the same call behaves differently by
provider. ``LLMConfig`` checks only that the schema is a JSON mapping; whether
it is closed is the vendor's to say, at request time.

Every object built here is **closed**: :func:`strict` lists each property in
``required`` and sets ``additionalProperties`` to false, and :func:`array` and
:data:`STRING` add no object of their own, so a schema composed only from
these is closed at every depth, which is the form both vendors ask for:

```python
from dataknobs_common.reply_schema import STRING, array, strict

line = strict({"sku": STRING, "amount": STRING})
invoice = strict({"number": STRING, "lines": array(line)})
# config = LLMConfig(provider="openai", model=..., response_schema=invoice)
```

**Closed is not the same as accepted.** A vendor also refuses keywords it does
not support, and which those are differs by vendor and changes over time. The
builders add none beyond ``type``, ``properties``, ``required``,
``additionalProperties`` and ``items``; a keyword a caller adds, such as a
length or count bound, is the caller's to check against the provider.

A part that is not built here (an integer, an ``enum``) is written as a plain
mapping. The builders do not check what keywords it uses, so a hand-written
object inside one is closed only if its author closed it.

**A built schema shares nothing with its parts**, or within itself. Each part
is rebuilt as data with :func:`~dataknobs_common.serialization.as_data`, so
every mapping in the result is a new plain ``dict`` and every sequence or set a
new ``list`` (a set's members sorted), and a part can be edited, or reused in
a second schema, without reaching what was built from it. A part holding a
value JSON cannot write (an object, a NaN, a container that holds itself) is
refused with :class:`~dataknobs_common.exceptions.SerializationError` naming
where it is, rather than by the vendor later.

The schema binds the reply's form only. What its values may say is the
caller's check, run on what comes back.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dataknobs_common.copying import FrozenMapping
from dataknobs_common.exceptions import SerializationError
from dataknobs_common.serialization import as_data

__all__ = ["STRING", "array", "strict"]

#: A string. Frozen, so no caller can change what every later schema gets.
STRING: Mapping[str, Any] = FrozenMapping({"type": "string"})


def _part(schema: Any, where: str) -> dict[str, Any]:
    """``schema`` rebuilt as a new plain ``dict``, every container in it new.

    Raises:
        TypeError: ``schema`` is not a mapping, such as the bare string
            ``"string"`` or ``FieldType.STRING`` where ``STRING`` was meant.
        SerializationError: ``schema`` holds a value JSON cannot write.
    """
    if not isinstance(schema, Mapping):
        raise TypeError(
            f"{where} must be a JSON Schema mapping, not {type(schema).__name__} ({schema!r})"
        )
    try:
        built: dict[str, Any] = as_data(schema)
    except SerializationError as error:
        raise SerializationError(
            f"{where}: {error}", context={**error.context, "part": where}
        ) from error
    return built


def array(items: Mapping[str, Any]) -> dict[str, Any]:
    """An array whose every item is ``items``.

    Args:
        items: The schema of each item.

    Returns:
        ``{"type": "array", "items": ...}``, holding a copy of ``items``.

    Raises:
        TypeError: ``items`` is not a mapping.
        SerializationError: ``items`` holds a value JSON cannot write.
    """
    return {"type": "array", "items": _part(items, "array items")}


def strict(properties: Mapping[str, Mapping[str, Any]], /) -> dict[str, Any]:
    """An object holding exactly ``properties``, every one required.

    The form OpenAI's strict mode requires, and closed as Anthropic requires:
    every property is listed in ``required``, in the order given, and
    ``additionalProperties`` is false. With no properties it admits only
    ``{}``.

    Properties are a mapping rather than keyword arguments, so any name can be
    written (``"class"``, ``"unit-price"``) and keywords stay free for options.

    Args:
        properties: Each property's name and its schema.

    Returns:
        The object schema, holding a copy of each property's schema.

    Raises:
        TypeError: ``properties`` is not a mapping, a name is not a string, or
            a property's schema is not a mapping.
        SerializationError: a property's schema holds a value JSON cannot
            write.
    """
    if not isinstance(properties, Mapping):
        raise TypeError(
            "strict properties must be a mapping of name to schema, not "
            f"{type(properties).__name__}"
        )
    built: dict[str, dict[str, Any]] = {}
    for name, schema in properties.items():
        if not isinstance(name, str):
            raise TypeError(
                f"strict property names must be strings, not {type(name).__name__} ({name!r})"
            )
        built[name] = _part(schema, f"property {name!r}")
    return {
        "type": "object",
        "properties": built,
        "required": list(built),
        "additionalProperties": False,
    }

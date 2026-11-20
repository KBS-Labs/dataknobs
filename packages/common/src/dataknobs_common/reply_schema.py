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
* **Anthropic** always enforces a schema, sent as ``output_config.format``, and
  requires the same closed objects.
* **Ollama** sends it as ``format`` and accepts any schema.

So a schema that leaves a property optional, or an object open, works under
Ollama and is refused with a 400 by OpenAI or Anthropic: the same call behaves
differently by provider. ``LLMConfig`` checks only that the schema is a JSON
mapping; whether it is closed is the vendor's to say, at request time.

Every object built here is closed. :func:`strict` lists each property in
``required`` and sets ``additionalProperties`` to false, and :func:`array` and
:data:`STRING` add no object of their own, so **a schema composed only from
these is closed at every depth**, and each of those providers accepts it:

```python
from dataknobs_common.reply_schema import STRING, array, strict

line = strict(sku=STRING, amount=STRING)
invoice = strict(number=STRING, lines=array(line))
# config = LLMConfig(provider="openai", model=..., response_schema=invoice)
```

A part that is not built here (an integer, an ``enum``) is written as a plain
mapping and copied as given. The builders do not check it, so a hand-written
object inside one is closed only if its author closed it.

**A built schema shares nothing with its parts.** Each builder copies what it
is given at every depth, so ``line`` above can be edited, or reused in a
second schema, without reaching ``invoice``. Every mapping in the result is a
plain ``dict`` and every sequence a ``list``, so it serializes with ``json``
whatever mappings it was built from.

The schema binds the reply's form only. What its values may say is the
caller's check, run on what comes back. A bound such as ``maxItems`` is the
caller's to add, by building a new schema around the old one rather than by
editing a shared part.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dataknobs_common.copying import FrozenMapping, thaw_structure

__all__ = ["STRING", "array", "strict"]

#: A string. Frozen, so no caller can change what every later schema gets.
STRING: Mapping[str, Any] = FrozenMapping({"type": "string"})


def _part(schema: Mapping[str, Any]) -> dict[str, Any]:
    """``schema`` as a new plain ``dict``, every container in it new too.

    Each part is thawed with its own memo, so two parts that are one object in
    the caller's hands are two objects in the result.
    """
    copied: dict[str, Any] = thaw_structure(schema)
    return copied


def array(items: Mapping[str, Any]) -> dict[str, Any]:
    """An array whose every item is ``items``.

    Args:
        items: The schema of each item.

    Returns:
        ``{"type": "array", "items": ...}``, holding a copy of ``items``.
    """
    return {"type": "array", "items": _part(items)}


def strict(**properties: Mapping[str, Any]) -> dict[str, Any]:
    """An object holding exactly ``properties``, every one required.

    The form OpenAI's strict mode and Anthropic require: every property is
    listed in ``required``, in the order given, and ``additionalProperties``
    is false. With no properties it admits only ``{}``.

    Args:
        **properties: Each property's name and its schema.

    Returns:
        The object schema, holding a copy of each property's schema.
    """
    return {
        "type": "object",
        "properties": {name: _part(value) for name, value in properties.items()},
        "required": list(properties),
        "additionalProperties": False,
    }

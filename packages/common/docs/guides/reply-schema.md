# Reply Schemas

`dataknobs_common.reply_schema` builds JSON Schemas for a model's reply that
are **closed by construction**: every object lists all of its properties in
`required` and sets `additionalProperties` to false.

```python
from dataknobs_common.reply_schema import STRING, array, strict

LINE = strict({"sku": STRING, "amount": STRING})
INVOICE = strict({"number": STRING, "notes": array(STRING), "lines": array(LINE)})
```

| Builder | Builds |
|---|---|
| `strict(properties)` | an object holding exactly `properties` (a mapping of name to schema): every one listed in `required`, in the order given, and `additionalProperties` false |
| `array(items)` | an array whose every item is `items` |
| `STRING` | `{"type": "string"}` (frozen) |

Import them from `dataknobs_common.reply_schema`. They are not exported from
the `dataknobs_common` package itself, where `STRING` would sit beside
`FieldType.STRING` (the text `"string"`, not a schema) and `array` would hide
the standard library's module under `import *`.

`strict` takes a mapping rather than keyword arguments, so any property name
can be written, `"class"`, `"unit-price"` or `"type"` included.

## Why closed

A schema is set on `LLMConfig.response_schema` (`dataknobs-llm`), and each
provider that supports one sends it. They differ on what they accept:

| Provider | Requires of every object |
|---|---|
| OpenAI, enforcing (strict mode: `LLMConfig.response_schema_strict`, on by default) | `additionalProperties: false`, and every property listed in `required` |
| Anthropic (always enforcing; sent as `output_config.format`) | `additionalProperties: false` |
| Ollama (sent as `format`) | nothing |

A schema with an open object therefore works under Ollama and is refused with
a 400 by OpenAI and Anthropic, and one with an optional property is refused by
OpenAI's strict mode, so a provider switched by configuration changes whether
the call works. `LLMConfig` checks only that the schema is a JSON mapping.

`strict` closes the object it builds, and `array` and `STRING` add no object
of their own, so **a schema composed only from these is closed at every
depth**. In `INVOICE` above, each line object is closed without its author
having to remember `additionalProperties` on the inner object.

### Closed is not the same as accepted

A vendor also refuses keywords it does not support, and which those are
differs by vendor and changes over time. The builders write only `type`,
`properties`, `required`, `additionalProperties` and `items`. Any other keyword
in a schema, such as a length or count bound, was added by the caller and is
the caller's to check against the provider they send it to.

The [`dataknobs-llm` guide](https://kbs-labs.github.io/dataknobs/packages/llm/guides/config-overrides/#asking-for-one-shape)
covers sending a schema: which providers support one, and what is refused
before the request.

## Other types

Only a string is provided. Write any other part as a plain mapping:

```python
ORDER = strict({
    "sku": STRING,
    "quantity": {"type": "integer"},
    "priority": {"type": "string", "enum": ["low", "high"]},
})
```

The builders do not check which keywords such a part uses, so a hand-written
*object* inside a built schema is closed only if its author closed it. Build
objects with `strict`.

A part must be a mapping: `strict({"sku": "string"})` and
`strict({"sku": FieldType.STRING})` raise `TypeError`, as does a property name
that is not a string.

## A built schema shares nothing

Each part is rebuilt as data with
[`as_data`](serialization.md), so every mapping in the result is a new plain
`dict` and every sequence or set a new `list`. A part can be reused in several
schemas, or edited after building, without any of them changing:

```python
line = strict({"sku": STRING, "amount": STRING})
invoice = strict({"number": STRING, "lines": array(line)})

line["properties"]["amount"]["pattern"] = r"^\d+$"
assert "pattern" not in invoice["properties"]["lines"]["items"]["properties"]["amount"]
```

A one-level copy would not hold here: `line["properties"]` would be one object
inside every schema built from `line`. Nor is anything shared *within* a
schema: a container that appears twice in a hand-written part is two objects
in the result.

What comes out is what `json` writes. A part handed in as a `MappingProxyType`
or a `FrozenMapping` becomes a `dict`, a tuple becomes a `list`, and a set
becomes a `list` with its members sorted, so the schema is the same on every
run. A part holding a value JSON cannot write (an object, a NaN, a container
that holds itself) is refused with `SerializationError`, naming the property
and the path inside it, rather than by the vendor at request time.

`STRING` is a `FrozenMapping`, so an edit to it raises `TypeError` rather than
changing every schema built afterwards.

## Bounds are the caller's

The schema binds the reply's **form**. What the values may say is the caller's
check, run on what comes back. A bound that varies per call is added by
building a new schema around the shared one each time, never by editing the
shared one:

```python
ROWS = array(strict({"label": STRING, "value": STRING}))

def bounded(rows: int) -> dict:
    return strict({"rows": {**ROWS, "maxItems": rows}})
```

`{**ROWS, ...}` copies only the top level, which is enough here because nothing
below it is edited, and `strict` then copies the result at every depth.
`maxItems` is one of the keywords a vendor may refuse (see
[Closed is not the same as accepted](#closed-is-not-the-same-as-accepted)), so
check the provider before sending it, or check the count on the reply instead.

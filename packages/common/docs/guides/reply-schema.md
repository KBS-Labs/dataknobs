# Reply Schemas

`dataknobs_common.reply_schema` builds JSON Schemas for a model's reply that
are **closed by construction**, so one schema is accepted by every provider
that can constrain its output:

```python
from dataknobs_common.reply_schema import STRING, array, strict

LINE = strict(sku=STRING, amount=STRING)
INVOICE = strict(number=STRING, notes=array(STRING), lines=array(LINE))
```

| Builder | Builds |
|---|---|
| `strict(**properties)` | an object holding exactly `properties`: every one listed in `required`, in the order given, and `additionalProperties` false |
| `array(items)` | an array whose every item is `items` |
| `STRING` | `{"type": "string"}` (frozen) |

The same three names are exported from `dataknobs_common`.

## Why closed

A schema is set on `LLMConfig.response_schema` (`dataknobs-llm`), and each
provider that supports one sends it. They differ on what they accept:

| Provider | Accepts |
|---|---|
| OpenAI | any schema as guidance; **only a closed one** when enforcing, which is strict mode (`LLMConfig.response_schema_strict`, on by default) |
| Anthropic | **only a closed one**, always enforced (sent as `output_config.format`) |
| Ollama | any schema (sent as `format`) |

*Closed* means every object lists all of its properties in `required` and sets
`additionalProperties` to false. A schema written with an optional property or
an open object therefore works under Ollama and is refused with a 400 under
OpenAI or Anthropic, so a provider switched by configuration changes whether
the call works. `LLMConfig` checks only that the schema is a JSON mapping;
closing it is the caller's job, and these builders do it.

`strict` closes the object it builds, and `array` and `STRING` add no object
of their own, so **a schema composed only from these is closed at every
depth**. In `INVOICE` above, each line object is closed without its author
having to remember `additionalProperties` on the inner object.

The [`dataknobs-llm` user guide](https://kbs-labs.github.io/dataknobs/packages/llm/guides/user-guide/#asking-for-one-shape)
covers sending a schema: which providers support one, and what is refused
before the request.

## Other types

Only a string is provided. Write any other part as a plain mapping:

```python
ORDER = strict(
    sku=STRING,
    quantity={"type": "integer"},
    priority={"type": "string", "enum": ["low", "high"]},
)
```

The builders copy such a part as given and do not check it. A hand-written
*object* inside a built schema is closed only if its author closed it, so
build objects with `strict`.

## A built schema shares nothing

Each builder copies what it is given **at every depth**, with
[`thaw_structure`](copying.md#freezing-a-structure). So a part can be reused
in several schemas, or edited after building, without any of them changing:

```python
line = strict(sku=STRING, amount=STRING)
invoice = strict(number=STRING, lines=array(line))

line["properties"]["amount"]["pattern"] = r"^\d+$"
assert "pattern" not in invoice["properties"]["lines"]["items"]["properties"]["amount"]
```

A one-level copy would not hold here: `line["properties"]` would be one object
inside every schema built from `line`. Every mapping in the result is a plain
`dict` and every sequence a `list`, so a part handed in as a
`MappingProxyType`, a `FrozenMapping` or with a tuple serializes with `json`
like any other.

`STRING` is a `FrozenMapping`, so an edit to it raises `TypeError` rather than
changing every schema built afterwards.

## Bounds are the caller's

The schema binds the reply's **form**. What the values may say is the caller's
check, run on what comes back. To bound a schema, such as a table with at most
one row per source row it was given, build a new schema around the shared one
for each call rather than editing the shared one:

```python
ROWS = array(strict(label=STRING, value=STRING))

def bounded(rows: int) -> dict:
    return strict(rows={**ROWS, "maxItems": rows})
```

`{**ROWS, ...}` copies only the top level, which is enough here because nothing
below it is edited, and `strict` then copies the result at every depth.

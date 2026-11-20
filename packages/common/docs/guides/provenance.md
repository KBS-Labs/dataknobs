# Provenance records

`dataknobs_common.provenance` records where a piece of content came from and
what has been done to it since: the sources it rests on, the tools and model
calls that made it, and each revision after. They are plain data, so content
written by a model can carry its provenance without dataknobs-bots. Bots
re-exports the same objects from `dataknobs_bots.artifacts.provenance`, and an
artifact's provenance is one of these records.

```python
from dataknobs_common.provenance import (
    LLMInvocation,
    ProvenanceRecord,
    RevisionRecord,
    SourceReference,
    ToolInvocation,
    create_provenance,
    now_iso,
)
```

The names are imported from the module; the `dataknobs_common` package does
not export them.

## Making a record

`create_provenance` names who or what made the content, and how. Every other
field is optional.

```python
from dataknobs_common.provenance import (
    LLMInvocation,
    SourceReference,
    ToolInvocation,
    create_provenance,
)

tool_chain = [
    ToolInvocation(
        tool_name="search_quotes",
        tool_version="1.2",
        parameters={"item": "stoneware mug", "max": 5},
    )
]

record = create_provenance(
    "llm:ollama:llama3.2",
    "llm_assisted",
    creation_context={"task": "Compare the mug quotes."},
    sources=[
        SourceReference(source_id="QUOTE-0117", source_type="row", confidence=0.9),
        SourceReference(source_id="QUOTE-0121", source_type="row"),
    ],
    tool_chain=tool_chain,
    llm_invocations=[
        LLMInvocation(purpose="write", model="llama3.2", invocation_id="llm-1"),
    ],
)
assert record.sources[0].source_id == "QUOTE-0117"
```

A field that holds several values takes any iterable, a list built up
beforehand included, and the record keeps a tuple of it. A single string is
refused where several values are held: `review_history="eval_1"` raises
`TypeError` rather than recording six one-letter ids.

## The records are frozen

A record handed to a store, a renderer or another task cannot change under it:

- every record is a frozen dataclass, so assigning to a field raises
  `dataclasses.FrozenInstanceError`;
- the fields holding several values are tuples;
- a mapping field (`ProvenanceRecord.creation_context`,
  `ToolInvocation.parameters`) keeps **a frozen copy** of the mapping it was
  given, a `FrozenMapping` built by `freeze_structure` (see
  [Copying](copying.md)). The caller's dict can go on changing without
  reaching the record, and whoever reads the record cannot change it: item
  assignment raises `TypeError`. Nested dicts become `FrozenMapping` too and
  nested lists tuples, so `record.creation_context["tags"]` reads back
  `("a", "b")` where `["a", "b"]` was given. Other values are shared, not
  copied, and a mapping that holds itself is refused with `ValueError`;
- `to_dict` returns new, plain containers, dicts and lists again, so what it
  returns can be written with `json.dumps` and edited without reaching the
  record.

So adding to a record makes a new one, with `dataclasses.replace`:

```python
import dataclasses

from dataknobs_common.provenance import RevisionRecord

revised = dataclasses.replace(
    record,
    revision_history=(
        *record.revision_history,
        RevisionRecord(reason="edited", triggered_by="user:buyer"),
    ),
)
assert len(revised.revision_history) == 1
assert record.revision_history == ()   # the record handed on is unchanged
```

`ArtifactRegistry.revise` in dataknobs-bots revises an artifact's provenance
the same way, so a superseded version keeps the provenance it had.

The records compare by value. `SourceReference`, `LLMInvocation` and
`RevisionRecord` are also hashable, so a set of sources works.
`ToolInvocation` and `ProvenanceRecord` hold a mapping, which cannot be
hashed, so they declare themselves unhashable: `hash()` raises `TypeError`
and `isinstance(record, collections.abc.Hashable)` is false.

## A revision of one part

A `RevisionRecord` describes a revision of the whole content by default.
Three optional fields describe a revision of one part of it:

| Field | Meaning |
|---|---|
| `part` | which part was revised, such as `line-2`; empty for the whole |
| `after` | the part's text after the revision: `""` where the part was removed, `None` where it was not recorded |
| `sources` | the ids of the sources the revised part kept |

```python
from dataknobs_common.provenance import RevisionRecord

edited = RevisionRecord(
    previous_version="Linden Ceramics is cheaper.",
    reason="edited",
    triggered_by="user:buyer",
    part="line-1",
    after="Linden Ceramics is cheaper per mug, but slower to deliver.",
    sources=["QUOTE-0117", "QUOTE-0121"],
)
removed = dataclasses.replace(edited, reason="rejected", after="")
assert removed.to_dict()["after"] == ""
assert "after" not in dataclasses.replace(edited, after=None).to_dict()
```

`LLMInvocation.invocation_id` is the matching addition on the other side: an
id a span of the content can name to say which call wrote it.

A `revision_id` is generated (`rev_` and 12 hex digits) when none is given,
both on construction and in `from_dict`. A caller that wants the same
decisions to give the same record passes its own.

## Serialised form

`to_dict` writes JSON-shaped data, each nested record by its own `to_dict`,
and `from_dict` reads it back from any mapping. An optional field that is unset
is left out:

| Record | Left out when |
|---|---|
| `SourceReference` | `source_location` or `excerpt` is `None`; `relevance` is empty |
| `ToolInvocation` | `tool_version` is `None` |
| `LLMInvocation` | `invocation_id` is empty |
| `RevisionRecord` | `part` is empty; `after` is `None`; `sources` is empty |

So a record that uses none of the part-level fields serialises exactly as a
record did before they existed, and a record stored then reads back
unchanged.

## Fields

### ProvenanceRecord

| Field | Type | Default | Description |
|---|---|---|---|
| `created_by` | `str` | `""` | Who or what created it (`"system:generator:quiz_gen_v1"`, `"user:jane"`, `"llm:<model>"`) |
| `created_at` | `str` | now | ISO 8601 creation timestamp |
| `creation_method` | `str` | `""` | How it was created (`"generator"`, `"wizard"`, `"manual"`, `"derived"`, `"llm_assisted"`) |
| `creation_context` | `Mapping[str, Any]` | empty | Parameters, configuration and other context; held as a frozen copy (`FrozenMapping`) of the mapping given |
| `sources` | `tuple[SourceReference, ...]` | `()` | References to the sources used |
| `tool_chain` | `tuple[ToolInvocation, ...]` | `()` | Tools used during creation |
| `llm_invocations` | `tuple[LLMInvocation, ...]` | `()` | Model calls made during creation |
| `review_history` | `tuple[str, ...]` | `()` | IDs of evaluations of the content |
| `revision_history` | `tuple[RevisionRecord, ...]` | `()` | Revisions made since |

### SourceReference

| Field | Type | Default | Description |
|---|---|---|---|
| `source_id` | `str` | required | ID of the source artifact, document, row or external reference |
| `source_type` | `str` | required | Category (`"artifact"`, `"document"`, `"vector_result"`, `"user_input"`, `"external"`, ...) |
| `source_location` | `str \| None` | `None` | URI, file path, corpus id or vector store reference |
| `relevance` | `str` | `""` | Why this source was used |
| `excerpt` | `str \| None` | `None` | The part of the source used |
| `confidence` | `float` | `1.0` | Relevance confidence (0.0 to 1.0) |

### ToolInvocation

| Field | Type | Default | Description |
|---|---|---|---|
| `tool_name` | `str` | required | Function or tool identifier |
| `tool_version` | `str \| None` | `None` | Version, if applicable |
| `parameters` | `Mapping[str, Any]` | empty | Input parameters; held as a frozen copy (`FrozenMapping`) of the mapping given |
| `timestamp` | `str` | now | ISO 8601 invocation time |

### LLMInvocation

| Field | Type | Default | Description |
|---|---|---|---|
| `purpose` | `str` | required | Why the model was called (`"decode_intent"`, `"write"`, `"extract"`, ...) |
| `model` | `str` | `""` | Model identifier used |
| `prompt_hash` | `str` | `""` | Hash of the prompt, for reproducibility (not the prompt) |
| `timestamp` | `str` | now | ISO 8601 invocation time |
| `invocation_id` | `str` | `""` | An id a span of the content can name to say this call wrote it |

### RevisionRecord

| Field | Type | Default | Description |
|---|---|---|---|
| `revision_id` | `str` | generated | Unique ID; `rev_` and 12 hex digits when not given |
| `previous_version` | `str` | `""` | The version, or the text, being revised |
| `reason` | `str` | `""` | Why the revision was made |
| `changes_summary` | `str` | `""` | What changed |
| `triggered_by` | `str` | `""` | Who or what triggered it (`"rubric_evaluation:eval_123"`, `"user:jane"`, `"system"`) |
| `timestamp` | `str` | now | ISO 8601 timestamp |
| `part` | `str` | `""` | Which part was revised; empty for the whole |
| `after` | `str \| None` | `None` | The part's text after; `""` where removed, `None` where unrecorded |
| `sources` | `tuple[str, ...]` | `()` | IDs of the sources the revised part kept |

### create_provenance()

```text
create_provenance(
    created_by, creation_method, *,
    created_at=None, creation_context=None,
    sources=(), tool_chain=(), llm_invocations=(),
    review_history=(), revision_history=(),
) -> ProvenanceRecord
```

Every argument after `creation_method` is keyword-only. `created_at` defaults
to now, in UTC; an unknown keyword raises `TypeError`.

### now_iso()

The current UTC time as an ISO 8601 string: the default for every
timestamp above.

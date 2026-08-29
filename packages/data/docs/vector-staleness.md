# When a vector is stale

A vector goes stale when the text that produced it stops being the text the
record would produce now. `VectorTextSynchronizer` re-embeds on exactly that
condition, and `ChangeTracker` reports it, so the two have to agree about what
"the text that produced it" means.

That agreement used to be unwritten, and it drifted. This page is the written
form.

## The rule

**The digest covers the string that was embedded** — not the field values, not
a canonical join of them. The bytes the embedder saw.

That is what makes "re-embed if and only if the input changed" true in both
directions. A vector is current when it was built from text identical to what
the record assembles today; it is stale otherwise; there is no third answer and
no heuristic.

## How the text is assembled

`dataknobs_data.vector.content` owns it, and both classes call it:

```python
from dataknobs_data.vector.content import assemble_source_text, compute_content_hash

text = assemble_source_text(record, ["title", "content"], separator=" | ")
digest = compute_content_hash(text)
```

`assemble_source_text` reads each named field in order, **drops falsy values**
rather than joining them as empty strings, and joins the rest on the separator.
`compute_content_hash` is md5 over the result — a same-or-different check, not
a security primitive.

## Two questions, and they have different answers

The assembly is described in the vector's metadata, and who may rely on that
description depends on which question they are asking.

| Class | Question | Authority |
|---|---|---|
| `ChangeTracker` | has the source changed since this was embedded? | the **record** |
| `VectorTextSynchronizer` | would I produce a different string now? | **its own** configuration |

A **reader** did not write the vector and has no standing to impose its own
assembly on it, so it reproduces the one the record describes:

```python
from dataknobs_data.vector.content import recompute_content_hash

current = recompute_content_hash(
    record,
    record.fields["embedding"].metadata,
    fallback_source_fields=my_own_fields,   # used only if the record does not say
)
is_stale = current is not None and current != record.fields["embedding"].metadata["content_hash"]
```

So a `ChangeTracker` never needs to be told the separator its
`VectorTextSynchronizer` used. Before the assembly was stored it was told
nothing, and hardcoded a space: a corpus synced with any other separator
reported **every** record outdated, forever, including records that had just
been synced and never edited.

A **writer** must do the opposite. It maintains the field, so its own
configuration decides, and the record's account of itself is history:

```python
from dataknobs_data.vector.content import current_content_hash

current = current_content_hash(record, my_source_fields, my_separator)
```

Deferring to the record here would make a synchronizer's own configuration
unchangeable. Re-point `text_fields`, or change `field_separator`, and every
record would go on matching the assembly it was written under — so `sync_all()`
would report nothing to do and the new configuration would never take effect,
silently and with no later sweep that would notice. The two functions exist
because collapsing them reads as a simplification and is a defect.

`sync_record` writes three keys into the vector field's metadata:

| Key | Holds |
|---|---|
| `content_hash` | the digest of the assembled text |
| `content_source_fields` | the field names that were assembled, in order |
| `content_field_separator` | what they were joined on |

Writing the assembly down is what stops the two configurations drifting apart
again. Passing a separator to both classes would have fixed the disagreement
that existed; it would not have removed the way to create the next one.

Both keys are validated on the way back in. They cross a persistence trust
boundary — whatever comes out of storage is not guaranteed to be the shape that
went in — so a field list that is not a list of names, or a separator that is
not a string, is logged and treated as absent rather than trusted.

## Records written before the assembly was stored

They carry `content_hash` alone. A reader falls back to its own configuration
for those, which is what they were digested under, so **no stored digest is
invalidated and nothing re-embeds on upgrade.**

They do not stay that way. When `sync_record` finds such a field current under
its own configuration, it writes the description before moving on — no
embedding, one write. The claim is true by construction rather than inferred
from the past: this synchronizer maintains the field and will re-embed it under
exactly that assembly from here on.

Without that step the upgrade would be one-way, and on a non-default separator
the two halves would deadlock: a tracker falls back to a space and reports the
whole corpus outdated, while the synchronizer correctly finds every record
current and so never rewrites one. Each half right, the corpus stuck.

## No stored digest means current

A vector field with no `content_hash` at all — hand-built, or written by
something other than this synchronizer — is treated as **current**. There is
nothing to compare against, and inventing a comparison would report every such
field stale on the first sweep.

That default assumes the digest survives storage, and whether it does is a
property of the backend. The rule: a backend that stores a field as a whole
object round-trips the digest; one that stores a bare column value does not.

| Backend | Vector field metadata | |
|---|---|---|
| memory, file (`json`), sqlite | round-trips | measured |
| file (`csv`, `tsv`, `parquet`) | **not preserved** | measured |
| elasticsearch | round-trips, for vector fields declared on the index | |

!!! warning "Flat file formats cannot detect staleness"

    A flat table has no column for a field's metadata, so those formats keep
    a vector's numbers and nothing else. A vector read back from one is a
    plain value rather than a `VectorField` and carries no digest — and with
    no digest to compare, an edited record is **never re-embedded**. Measured
    on the same corpus, one edit, one sweep:

    | Format | `sync_all()` after an edit |
    |---|---|
    | `json` | `updated=1`, one embedding |
    | `csv` | `updated=0`, no embeddings |

    This is silent. Use `force=True` on those formats, or a backend that
    preserves field metadata, if the corpus is edited after it is embedded.

`ChangeTracker.get_outdated_records` backfills the digest for a field that has
none rather than queueing it: it computes what the digest would be now, stores
it, and moves on. A record it cannot write — one with no id, or an id nothing
is stored under — is reported outdated instead, since the next sweep would
otherwise arrive at the same place.

## What each entry point does

| Call | Re-embeds |
|---|---|
| `sync_record(record)` | every registered vector field whose digest no longer matches |
| `sync_record(record, force=True)` | every registered vector field |
| `sync_record(record, fields=[...])` | as above, restricted to those fields |
| `sync_all()` | the same rule, over every record |
| `sync_on_update(id, old, new)` | only the vector fields fed by a source that actually changed |

`force=True` is how you ask for an embedding regardless — after a model change,
say. Without it, a sweep over an unchanged corpus costs nothing. One case it
does not cover: a record whose source fields are all empty assembles no text,
and there is nothing to embed, so its existing vector is left in place.

A field is *registered* if either the database schema declares it with a
`source_field`, or it is the `vector_field` of a synchronizer constructed with
`text_fields=`. Both sources are registered together, so every entry point sees
every field however it was declared. Where they name the same vector field,
`text_fields=` **replaces** what the schema said about it rather than adding to
it — a source the schema named and `text_fields` does not stops feeding that
vector.

## When the write does not land

`sync_record` computes vectors onto the record you hand it, then persists them
— and the record is persisted **whole**, so the record you hand it must be
complete. Building one out of a partial update and passing that replaces the
stored record with those fields alone.

The returned `success` reports whether the write landed. It is `False` when
there is no id to write under, and equally when nothing is stored under the id
the record carries — a `Record` falls back to an `id` field in its data, so it
can carry one without ever having been written. `updated_fields` still names
the fields in both cases, because the vectors really are on the object you
hold; what is false is the claim that they were stored.

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

That default rests on two assumptions, and each has been false somewhere: that
the writer recorded a digest, and that storage gave it back.

### Every writer records one

A writer that does not makes its whole output permanently exempt — and
silently, since the sweep that skips it reports success. Three of them did not.

| Writer | Where the description lives |
|---|---|
| `bulk_embed_and_store`, sync and async | on the `VectorField` |
| `VectorSyncMixin.sync_vectors_with_text` | on the `VectorField` |
| `VectorTextSynchronizer` | on the `VectorField` |
| `VectorMigration` | on the `VectorField` |
| `IncrementalVectorizer` | a `{field}_metadata` sidecar |

The async Postgres backend and `VectorMigration` each built their own
`VectorField` instead of the shared one and omitted the digest; both now route
through `attach_vector_field`, which is the one place that field is built for
text this package embedded. `IncrementalVectorizer` stores a plain list rather
than a `VectorField`, so it describes the vector in a sidecar record field —
a different place, not a different contract, and it now keeps the same three
keys there. Both lanes ask one function whether the digest still matches.

The sidecar is written whether or not a model was named. The digest is the half
that does not depend on one, and gating the whole description on `model_name`
left an unnamed vector undescribed.

### And every writer records which model

A digest answers "is this the text that produced the vector?". It cannot answer
"was it produced by the model now in use?" — identical text through two models
gives one digest and two incompatible vector spaces, so a swap is invisible to
a check that reads only the text.

`model_name` is that second key, and `TextEmbedder.model_id` is what supplies
it: pass `embedder=` and the name written beside the vector comes from the
thing that produced it rather than from a parameter a caller keeps in step by
hand. Where the vectors go decides which spelling:

| Writer | Where the identity lives |
|---|---|
| `bulk_embed_and_store` (database) | the `VectorField`'s `model_name` |
| `VectorSyncMixin.sync_vectors_with_text` | the `VectorField`'s `model_name` |
| `VectorTextSynchronizer`, `VectorMigration` | the `VectorField`'s `model_name` |
| `IncrementalVectorizer` | the `{field}_metadata` sidecar |
| `bulk_embed_and_store` (`VectorStore`) | each vector's metadata, under the key `add_records` uses |

An explicit `model_name=` wins over the embedder's own, so a caller who said
what they meant is not overridden. `model_version` is never defaulted from an
embedder anywhere: a `TextEmbedder` carries an identity and no version.

A stored `None` is deliberately not a mismatch, on either key. A vector written
before anything recorded a name says nothing about its model, and reading that
silence as evidence of a *different* one would re-embed every pre-seam corpus
on the first sweep after upgrading — the same trade `content_hash` makes one
section above.

### The digest survives storage

Whether it does is a property of the backend.

| Backend | Vector field metadata | |
|---|---|---|
| memory, file (`json`), sqlite | round-trips | measured |
| file (`csv`, `tsv`) | round-trips | measured |
| file (`parquet`) | round-trips | shares the flat-format path; `pyarrow` is an optional extra, so a default test run does not measure it |
| elasticsearch | round-trips, for vector fields declared on the index | measured, against a live cluster |

### How a flat format carries it

A flat table has one cell per field and nowhere to put that field's `type` or
`metadata`, so `csv`, `tsv` and `parquet` used to reduce a field to its bare
`value`. A `VectorField` went in and a plain `Field` holding a list of numbers
came back — carrying no digest, and so judged current forever. That was silent,
and it is fixed: the reduction is now conditional.

| The field | The cell |
|---|---|
| a plain scalar value, no metadata | the bare value |
| anything more — a vector, or any field carrying metadata | the whole field dict, as JSON |

So a CSV of ordinary records still opens in a spreadsheet as ordinary columns,
which is the reason to ask for one, and a vector column holds a JSON object that
`Record.from_dict` reconstructs into the same `VectorField` that was written.

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

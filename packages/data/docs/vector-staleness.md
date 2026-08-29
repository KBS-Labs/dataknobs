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

## The assembly is stored, so a reader need not be configured

`sync_record` writes three keys into the vector field's metadata:

| Key | Holds |
|---|---|
| `content_hash` | the digest of the assembled text |
| `content_source_fields` | the field names that were assembled, in order |
| `content_field_separator` | what they were joined on |

The last two are what make the contract hold between classes. A reader
reproduces the text from **what the record carries**, not from how the reader
itself was configured:

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
been synced and never edited. Both callers in the tree sat on the default
separator, one of them explicitly, which is why that went unseen.

Writing the assembly down is also the reason the two constructor arguments
cannot drift apart again. Passing a separator to both classes would have fixed
the disagreement that existed; it would not have removed the way to create the
next one.

## Records written before the assembly was stored

They carry `content_hash` alone. `recompute_content_hash` falls back to the
caller's own configuration for those, which is what they were digested under, so
**no stored digest is invalidated and nothing re-embeds on upgrade.**

## No stored digest means current

A vector field with no `content_hash` at all — hand-built, or written by
something other than this synchronizer — is treated as **current**. There is
nothing to compare against, and inventing a comparison would report every such
field stale on the first sweep.

`ChangeTracker.get_outdated_records` backfills the digest for these rather than
queueing them: it computes what the digest would be now, stores it, and moves
on. The record is then judged normally from the next sweep onwards.

## What each entry point does

| Call | Re-embeds |
|---|---|
| `sync_record(record)` | every registered vector field whose digest no longer matches |
| `sync_record(record, force=True)` | every registered vector field, unconditionally |
| `sync_all()` | the same rule, over every record |
| `sync_on_update(id, old, new)` | only when a source field actually changed |

`force=True` is how you ask for an embedding regardless — after a model change,
say. Without it, a sweep over an unchanged corpus costs nothing.

A field is *registered* if either the database schema declares it with a
`source_field`, or it is the `vector_field` of a synchronizer constructed with
`text_fields=`. Both sources are registered together, so every entry point sees
every field however it was declared.

## Records with no id

`sync_record` computes vectors onto the record you hand it, then persists them.
A record that was never stored has no id to persist under, so nothing is
written — and the returned `success` is `False` while `updated_fields` still
names the fields, because the vectors really are on the object you hold. Store
the record first if you want the write.

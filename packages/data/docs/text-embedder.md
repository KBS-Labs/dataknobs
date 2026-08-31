# Turning text into vectors

Everything in this package that embeds text now names one type. Before, it
named eight, and none of them was the shape an LLM provider returns — so
every consumer wiring a provider to a vector path wrote an adapter, and wrote
a different one at each of twenty-five call sites.

This page is the contract those sites now share.

## The protocol

```python
from dataknobs_data.vector import TextEmbedder
```

Three members, and each is narrower than what it replaced:

| Member | Contract |
|---|---|
| `async embed(texts) -> list[list[float]]` | one vector per input text, in order; an empty batch is an empty result, not an error |
| `dimensions -> int` | the length of every vector `embed` returns |
| `model_id -> str` | stable identity of the model producing them — across processes, not merely within one |

`isinstance(x, TextEmbedder)` works and checks that the three members are
present. It does not check their signatures, and `issubclass` is unavailable
(a protocol carrying non-method members cannot support it). Treat it as a
smoke test.

### How an implementation proves it is one

Nothing inherits this protocol. An adapter lives in whichever package holds
the thing being adapted, and two of the four are in packages `data` cannot
import, so every implementation matches the protocol structurally and says so
in a docstring. A docstring is checked by nobody: it is true when written and
free to stop being true when a signature drifts, and nothing raises until a
consumer's call fails somewhere that names neither the class nor the protocol.

So each implementation states it where the type checker can read it:

```python
if TYPE_CHECKING:
    from dataknobs_data.vector.embedding import TextEmbedder

    def _satisfies_text_embedder(x: MyEmbedder) -> TextEmbedder:
        return x
```

One type-check, no runtime import, and strictly stronger than the `isinstance`
above — a wrong return type, or an `embed` taking one text instead of a batch,
passes that and fails this. Write it for any embedder you implement.

### Why batch-only

`AsyncLLMProvider.embed` is arity-polymorphic: `str` in gives `list[float]`
out, `list[str]` in gives `list[list[float]]` out. That is convenient at a
call site and awful as a contract, because every consumer has to narrow the
union — and much of the fragmentation this replaced *was* that narrowing done
six different ways. One text is:

```python
vector = (await embedder.embed([text]))[0]
```

### Why async-only

Every real embedding source is network or GPU I/O, and running that on the
event loop stalls every other task sharing it. A synchronous caller reaches an
embedder through [`SyncTextEmbedder`](#reaching-an-embedder-from-synchronous-code),
not through a second protocol — a synchronous twin would put back the second
shape this exists to remove.

### Why `list[list[float]]` and not `list[np.ndarray]`

It is what `AsyncLLMProvider.embed` already returns, so the adapter spanning
the two packages performs no conversion — which is the seam's whole job. It is
also the shape `sources/processing.py` and `sources/cluster_index.py` each
arrived at independently, the only one anything in this tree converged on
twice. And `numpy` is undeclared in `dataknobs-llm`, so putting `np.ndarray`
in a public `llm` signature would force a dependency declaration to buy a
conversion at the other end. `np.asarray(...)` at a `data`-side consumer is one
call, on the side where numpy is declared.

### Why it carries its own identity

The database method `bulk_embed_and_store(records, text_field, …)` takes
`embedding_fn`, `model_name` and `model_version` as three independent
parameters, and trusts the caller to keep the name in step with the function.
The name is the *staleness key* — it is what a later reader uses to decide
whether a stored vector is still comparable — so a mismatch is discovered only
by something that trusts it.

`VectorStore.bulk_embed_and_store(texts, …)` is a different method that shares
the name: it stores bare vectors rather than records, so the identity goes into
each vector's metadata under the same `model_name` and `model_version` keys
that `add_records` copies off a `VectorField`. Those two are the store's only
entry points, and until they agreed, whether a stored vector could be judged
against a model swap depended on which of them put it there.

`model_id` removes that class of error. Pass `embedder=` and `model_name`
defaults to the identity of the thing that actually produced the vectors:

```python
await db.bulk_embed_and_store(records, "body", embedder=embedder)
# the stored VectorField's model_name is embedder.model_id

await store.bulk_embed_and_store(texts, embedder=embedder)
# each vector's metadata["model_name"] is embedder.model_id
```

`model_version` is never defaulted from an embedder, on any path. A
`TextEmbedder` carries an identity and no version, so filling that key from one
would write a value nothing produced.

An explicit `model_name=` still wins, so a caller who said what they meant is
not overridden.

Defaulting the key is only half of it. `VectorTextSynchronizer` compares
`model_name` — under `SyncConfig.track_model_name`, alongside the older
`model_version` comparison — so a corpus embedded by one model and swept by a
synchronizer configured for another reports stale and is re-embedded. Without
that half, `model_id` would be recorded on every vector and consulted by
nothing, which describes the vector without protecting anyone.

A stored `None` is deliberately not a mismatch. A vector written before
anything recorded a name carries no information about its model, and treating
that absence as evidence of a different one would re-embed every pre-seam
corpus on the first sweep after upgrading.

## Getting one

`dataknobs-llm` supplies the implementation, because the dependency runs that
way and only that way — `dataknobs-data` cannot import `dataknobs-llm`.

```python
from dataknobs_llm import create_text_embedder

embedder = await create_text_embedder(
    {"embedding": {"provider": "ollama", "model": "nomic-embed-text"}}
)
```

There is deliberately **no new config type**: an embedder config *is* an
`LLMConfig`. `create_text_embedder` wraps `create_embedding_provider`, which
already accepts a typed config or any of the dict forms and already forces
`mode=embedding`.

Where the config declares no vector width, pass one — otherwise `dimensions`
raises until the first `embed` has been observed, which is the section below:

```python
embedder = await create_text_embedder(config, dimensions=1536)
```

To adapt a provider you already hold:

```python
from dataknobs_llm import LLMProviderEmbedder

embedder = LLMProviderEmbedder(provider)
```

`LLMProviderEmbedder` does not inherit `TextEmbedder` — it satisfies the
protocol structurally, and imports it under `TYPE_CHECKING` to assert that
statically. See [How an implementation proves it is one](#how-an-implementation-proves-it-is-one).

### `dimensions` is answered, never guessed

From what was declared (the constructor argument, else the provider's
configured `dimensions`), and otherwise from what was *observed* on the first
`embed`. Never by probing, because a probe is a network round trip and this is
a property callers read freely. If nothing declared one and nothing has been
embedded, it raises rather than guessing — a wrong width is enforced later by a
vector store, and a dimension mismatch discovered at write time names the
store rather than the embedder that was actually misconfigured.

A *declared* width is checked against the first batch rather than trusted,
because the two can disagree and nothing else in the stack would notice.

## Using one

### On the write paths

`embedder=` is keyword-only everywhere, so it cannot be mistaken by position
for the `embedding_fn` it sits beside:

```python
ids = await store.bulk_embed_and_store(texts, embedder=embedder)
count = await db.sync_vectors_with_text(records, ["title", "body"], embedder=embedder)
```

The long-running paths take it too — a sweep, a migration, a background
vectorizer, a dedup pass:

```python
sync = VectorTextSynchronizer(database=db, text_fields=["body"], embedder=embedder)
vectorizer = IncrementalVectorizer(database=db, text_fields="body", embedder=embedder)
migration = VectorMigration(source_db=db, text_fields=["body"], embedder=embedder)
checker = DedupChecker(db=db, config=DedupConfig(semantic_check=True),
                       vector_store=store, embedder=embedder)
```

`VectorTextSynchronizer` is the one where this closes a loop rather than saving
a line. It writes `model_name` and reads it back, so passing an embedder is what
makes those two the same fact — before, a caller with an embedder named the
model twice, once by passing `embed` and once by passing `model_name=`, with
nothing checking that the two agreed.

`DedupChecker` closes the same loop, and only for `embedder=`. `register()`
records the embedder's `model_id` beside each vector and `check()` compares it,
reporting a disagreement on `DedupResult.mismatched_model_ids`. An
`embedding_fn` carries no identity to record, so that lane writes no key and
gets no comparison — see [Content Deduplication](dedup.md).

`sync_vectors_with_text` closes the same loop for a caller that owns its own
records. It writes the identity and compares it, so a sweep with a second
embedder re-embeds rather than reporting a corpus current on the strength of
text that did not change — the digest cannot see a model swap, because
identical text through two models gives one digest and two incompatible vector
spaces.

Pass **one** of `embedder` and `embedding_fn`. Passing neither raises; so does
passing both — resolving that by precedence would mean one of the two silently
does not run, and the caller cannot tell which.

Two of the classes above permit **neither**, and that is deliberate rather than
inconsistent. `VectorMigration` can add the schema field without embedding, and
`DedupChecker` does exact-hash matching with no semantic pass at all, so
demanding a source at construction would refuse a supported use.
`VectorTextSynchronizer` and `IncrementalVectorizer` exist only to embed, so
they raise at construction — finding out on the first record means failing
after a query, a batch and a partial write.

What the two permissive classes do *later* differs, and the difference is worth
knowing before you rely on either. `VectorMigration` raises where a vector is
actually produced: `add_vectors_to_existing` refuses with "an embedding source
is required", while `run()` migrates records without vectors, which is the use
the permission exists for. `DedupChecker` never raises — with
`semantic_check=True` and no source it silently degrades to exact-hash
matching, storing no vector and reporting `"unique"`. See
[Content Deduplication](dedup.md).

The callable path is not deprecated, and no call site of it has to change. An
untyped `embedding_fn` still has to be classified before it is called, and
`vector/embedding_fn.py` is now the one place that happens —
`call_embedding_fn` for a single text, `call_embedding_fn_batch` for a corpus,
over one shared resolver. An *adopted* site has nothing left to classify,
because a `TextEmbedder` is async by declaration.

Two things that dispatch guarantees, which matter to anyone still passing a
callable. A **synchronous** one is offloaded with `asyncio.to_thread` rather
than run on the event loop, at both arities — embedding is CPU- or
network-bound, and a corpus embedded inline stalls every other task on the loop
for its whole duration. And the *result* is re-examined even when the callable
classified as synchronous, because a plain `def` that returns a coroutine is
genuinely synchronous and still hands back something that has to be awaited
before it is a vector.

### Caching

```python
from dataknobs_data.vector import CachedEmbedder
from dataknobs_llm.llm.providers.caching import MemoryEmbeddingCache

embedder = CachedEmbedder(embedder, MemoryEmbeddingCache())
```

The key is `(inner.model_id, text)`, and the model half is not optional. A
cache keyed on text alone does not fail loudly after a model swap: every lookup
*succeeds*, and hands back vectors from a model no longer in use, in a vector
space the new one knows nothing about. Nothing raises and nothing is logged —
the similarities are simply wrong.

`CachedEmbedder` takes any `VectorCache`, which is a two-method port:
`get_batch` and `put_batch`. Both of `dataknobs-llm`'s shipped caches satisfy
it structurally, where they already are. Only the texts that miss reach the
inner embedder, and they reach it in one batch; a text repeated within a batch
costs one embedding and fills both output slots.

`model_id` and `dimensions` forward unchanged, so a vector stored through a
cached embedder carries the same staleness key as one stored without it — a
caller must not be able to tell from the metadata that a cache was in the path.

### Reaching an embedder from synchronous code

Five sites in this package embed inside a plain `def`: `Query.near_text`,
`VectorField.from_text`, and the three synchronous `bulk_embed_and_store`
lanes. None of them can await, and none of them grew an `embedder` parameter.

```python
from dataknobs_data.vector import SyncTextEmbedder

with SyncTextEmbedder(embedder) as sync:
    query.near_text("some text", sync.embed_one)
    store.bulk_embed_and_store(records, "body", embedding_fn=sync.embed)
```

`sync.embed` satisfies `Callable[[list[str]], np.ndarray | list[list[float]]]`
and `sync.embed_one` satisfies `Callable[[str], np.ndarray | list[float]]` —
between them, the shapes those five sites declare. So the sync lanes need no
new parameter to reach the seam.

The list arm of each union is not an accommodation added for this class. Those
sites hand the result to `pair_records_with_vectors`, which requires only
"something indexable and sized", and `near_text` hands it to `Query.similar_to`,
which has always declared `np.ndarray | list[float]`. The parameters previously
said `np.ndarray` alone, which understated what they had always accepted — so
the snippet above was correct at runtime and an `arg-type` error under `mypy`
at three of the sites. The annotations now say what the code does.

`SyncTextEmbedder` holds one `SyncLoopBridge`: a private event loop on a daemon
thread, which makes it callable from plain sync code *and* from inside a
running loop, without the `asyncio.run` / `run_until_complete` deadlock. That
costs one daemon thread for the object's lifetime, so build one and keep it
rather than one per call.

**It does not deadlock; it does block.** Called from inside a coroutine, the
calling thread waits on the bridge's result for the whole embedding, so every
other task on the caller's loop is stalled for a network round trip — a cost
no exception reports and no single-request test shows. From async code, await
the embedder directly; this class is for the five `def` sites that cannot.

`close()` is for deterministic teardown, and it closes only the bridge — the
wrapped embedder was handed in already built and is not its to close. Dropping
one without closing it emits a `ResourceWarning` naming the loop thread, and
the bridge tears itself down; before that it could not, because the live loop
thread held a reference back to the bridge that kept it permanently alive.

It is **not** a `TextEmbedder`, and this is the one place the protocol's
runtime check will tell you otherwise. `isinstance(sync, TextEmbedder)` answers
`True`, because a runtime-checkable protocol checks that the three members are
present and nothing about their signatures — and `SyncTextEmbedder.embed` is a
plain `def` where the protocol's is `async def`. That is precisely the limit
the protocol's own docstring names, met by the one class in this package shaped
to trip it.

Handing one where an embedder is expected therefore fails on the `await`, not
on the check: `TypeError: object list can't be used in 'await' expression`.
Loud and immediate, so nothing stores a non-vector — but it is the annotation,
not the `isinstance`, that stops the mistake being made.

## Testing against one

```python
from dataknobs_data.testing import DeterministicEmbedder

embedder = DeterministicEmbedder(dimensions=8, model_id="v1")
```

Stable across processes and runs — two instances agree — and distinct texts
land near-orthogonal, so a *ranking* can be asserted rather than merely a
distance. Changing `model_id` changes the vector space, which is what makes it
usable for testing a staleness path: without that, no test could distinguish
"invalidated and re-embedded" from "served the old vector".

It does not reuse the older `text_embedding` helper, which draws every
component from `[0, 1)` — already documented on `chroma_embedding_function` as
making it unusable for asserting a ranking.

## Related

- [When a vector is stale](vector-staleness.md) — what `model_id` is the key to
- [Hybrid search](hybrid-search.md)

# Changelog

All notable changes to Dataknobs packages will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Release - 2026-09-03

### dataknobs-llm [0.10.0]

#### Added
- `fsm` extra — `dataknobs-fsm` backs `fsm_integration` and `ConversationFlowAdapter` but was declared in no dependency list and no extra, not even `[all]`
- `mock_tool_arguments()` (`dataknobs_llm.testing`) — deterministic stand-in tool-call arguments derived from a tool's JSON schema, honoring `enum`, nested `properties`, array `items` and `required_only=`
- `rag_config_from_dict()` (`dataknobs_llm.prompts`) — one stated home for the recognized `RAGConfig` keys; `add_message(rag_configs=...)` parses through it rather than forwarding what it was handed
- declare `dataknobs-structures`; `Tree` backs `ConversationState` at module scope, so even a bare `import dataknobs_llm` needed it, and it was reaching installs only through `dataknobs-utils`

#### Changed
- `EchoProvider.set_responses()` accepts any sequence — the parameter was an invariant `List`, so a `list[str]` was a type error at every call site

#### Removed
- **Breaking:** the deprecated `function_call()` provider API and `LLMResponse.function_call`, with the `adapt_raw_functions()` helpers, the legacy base shim, and the prompt-based tool-calling fallback that lived only on that path — deprecated since v0.4.0 in favor of `complete(tools=...)` and `response.tool_calls`. `LLMMessage.function_call` (the wire format) and `LLMConfig.function_call` (the request parameter) are unaffected
- `EchoProvider`'s schema-derived mock arguments, returning as `mock_tool_arguments()` — a pure function of the schema, so any provider double can use it

#### Fixed
- conversation flows never ran a single state, in any release — the adapter emitted a `functions` key `FSMConfig` does not define, then called `process_async` on the *synchronous* facade; nine further failures sat behind those, each reachable only once the one in front of it was removed
- flow runs drive `AsyncSimpleFSM` and close it, instead of blocking the event loop for their whole duration
- the state transform and arc condition read the engine's `FunctionContext` as what it is, and call a prompt-builder method that exists — a render failure now fails the run instead of being delivered to the conversation as assistant content reading `[Error in state <name>]`
- a failed flow raises `OperationError` instead of reporting success and handing back the caller's own input; a tripped loop guard reports its own `stop_reason`; a condition that *raised* is no longer read as a condition that said "no"
- `ConversationFlowAdapter(llm=...)` reaches `LLMClassifierCondition`, and the adapter's internal keys — `_llm_provider` and the live credential it can hold, `_force_end`, `_error` — no longer reach prompt templates
- `get_execution_summary()["current_state"]` names the state the flow ended in, not the first one; `execute_flow()` builds nodes at all, having constructed `ConversationNode` with parameters it does not accept
- an FSM tool or `DeterministicTask` whose callable is an *object* with an `async def __call__` now runs and is awaited — `iscoroutinefunction` answers `False` for that shape, so it was reported successful with an un-awaited coroutine as its value and its body never ran
- synchronous FSM tools run on a worker thread rather than inline on the event loop
- `LLMCaller` and `EmbeddingGenerator` require an `AsyncLLMResource` and say so by name — both guarded on the synchronous base and then used the async API only its subclass provides
- `LLMCaller` names a broken usage contract instead of reporting `tokens_used: None`
- `PromptBuilder` nested variables resolve as a path of keys — a dotted entry such as `"user.name"` reached `str.format()` under its dotted key, so every such template raised
- `import dataknobs_llm.conversations` no longer requires `dataknobs-fsm`, which had put `ConversationManager` out of reach on every base install
- the buffered OpenAI path returns tool calls — `adapt_response` built its `LLMResponse` without them, silently disarming `_analyze_response` and `_warn_if_truncated` on that path
- unreadable tool-call arguments are settled once on the adapter base, and `ToolCall` normalizes `parameters` however an instance was built — six sites across four providers had answered differently, and a call read back from storage kept whatever was stored
- `get_total_cost()` and `get_cost_by_branch()` stop at a leaf, where `Tree.children` is `None` rather than empty — cost tracking had never recorded a cost, the walk raising on the first leaf of every branch inside a handler that logged and continued
- `stream_complete()` reports a `reset()` arriving mid-turn as the reason `complete()` has always given, rather than an `AttributeError`, and no longer hands `None` to post-response middleware that declares a `ConversationState`

#### Documented
- the conversation-flow pages describe the shipped API — seven published pages taught a `FlowState` shape, a `ConversationManager.create(flow=...)` parameter and an `execute_flow()` returning a response, none of which has ever existed. They now state the thing the old pages inverted: a flow state renders its prompt rather than calling the LLM. Every code block on the two example pages is executed as written before publication

### dataknobs-bots [0.13.0]

#### Changed
- raised the `dataknobs-llm` floor to `>=0.10.0`

## Release - 2026-09-02

Four changes run across the workspace in this release.

**A callable object can be `async`, and almost nothing in the workspace asked.**
`inspect.iscoroutinefunction` answers for *functions*, and reports a callable
object whose `__call__` is an `async def` as synchronous — which is how anything
stateful is written. `dataknobs-common` now exports `is_async_callable`, plus
`run_callback` and `run_callback_off_loop`, which judge the *result* and so also
catch a plain `def` returning a coroutine. Every dispatch that had answered the question for
itself now routes through them: fourteen in `dataknobs-data`, nine modules in
`dataknobs-fsm` spelling it five different ways, and `CallbackRegistry.fire` in
common. Calling an async callable without awaiting raises nothing — the
coroutine is truthy and non-`None` — so all of them failed in silence: a stream
filter's predicate answered `True` unconditionally and stopped filtering, a
transform yielded the coroutine as the transformed record, an arc condition
never ran. On upgrade, an async callback that was silently doing nothing now
runs, and on the paths that offload a synchronous one now runs on a **worker
thread** rather than the loop thread.

**One shape for turning text into vectors, and an identity that travels with
them.** `dataknobs-data` declares `TextEmbedder` — batch-only, async-only,
carrying `dimensions` and `model_id` — and `dataknobs-llm` ships
`LLMProviderEmbedder` and `create_text_embedder` satisfying it. Eight
incompatible spellings were in use in `data` alone, none matching what
`AsyncLLMProvider.embed` returns, so every consumer wiring a provider to a
vector path wrote its own adapter — differently at each of 25 call sites.
`embedder=` now appears on the write paths and on the four classes that embed a
corpus over time; the `EmbedFn` / `BatchEmbedFn` / `ClusterEmbedFn` aliases are
**gone with no deprecation cycle**. What the protocol adds over a bare callable
is the identity: `model_id` is the staleness key, named once rather than twice
with nothing checking the two agreed. Its absence is a family of defects: vectors stored
naming no model, a synchronizer comparing a field the embedder path never sets
so swapping embedders left every old vector current forever, dedup scoring
across two vector spaces and answering `unique`. In `dataknobs-llm`,
`LLMConfig.dimensions` was read by one provider of five, so a config asking for
512-dimensional vectors received 3072.

**A registry gate that rejected the shape it exists to admit, and admitted the
shape it exists to exclude.** `PluginRegistry.validate_type` promises that a
structurally wrong backend is caught at registration rather than at use, and it
held in neither direction. Given a Protocol carrying a property, `issubclass`
refuses the whole call rather than the offending member, so the gate raised
before judging anything — on the conforming class as readily as on a wrong one,
which made the constraint strictly worse than its own absence. `dataknobs-llm`
met this on its own documented extension point. In the other direction a
`@runtime_checkable` check asks only whether a member is present, so a Protocol
and its async twin satisfy each other and a registry gated on the synchronous
half accepted an `async def` implementation — four such pairs ship here.
Separately, the four factory-invocation sites each decided independently whether
to await a factory's result: a synchronous `get()` handed back an un-awaited
coroutine and *cached* it, so the first caller awaited it and every later one
got the same exhausted object. All four now share one seam, and both gates one
implementation.

**A vector store says what it can do, rather than being asked to prove it by
downcast.** `dataknobs-common` adds `Capability.VECTOR_PERSIST`,
`VECTOR_INDEX_TUNING` and `VECTOR_DOCUMENT_API`, and `supports_capability`, the
non-raising half of the guard pair. `VectorStore` implements
`CapabilityContract`, and `save()`, `load()` and `create_index()` are declared
on the abstraction so the answer can be acted on without an `isinstance`
downcast. `VECTOR_PERSIST` is instance-scoped — only `MemoryVectorStore` and
`FaissVectorStore` given a `persist_path` advertise it, and both now raise where
they previously answered a request to persist with a successful-looking no-op.
The case to look for is a `hasattr(store, "save")` probe: it now answers `True`
for every backend, so a guard written that way stops guarding.

### dataknobs-common [3.2.0]

#### Added
- `is_async_callable` — "will calling this produce an awaitable?", answered for callable objects and not only functions, as a `TypeGuard` so it narrows in the `True` branch. Two shapes the obvious implementation gets wrong are handled: a `functools.partial` around a callable object, whose `__call__` is a C dispatcher that answers about the wrong object, and a class, which is callable and returns an instance however its `__call__` is declared. Non-callables answer `False` rather than raising, because callers ask about arbitrary configured values
- `run_callback` and `run_callback_off_loop` — call a consumer's callback and await it if it turned out to be awaitable, the second keeping a synchronous callback off the event loop via `asyncio.to_thread`. The pair states a choice that was previously left unmade: a thread hop costs real microseconds and moves the callback off the loop thread, which is right where the callback is the consumer's chance to *do* something and wrong for a predicate or a per-record transform. Extracted rather than invented — `CallbackRegistry.fire_async` had carried the judgement inline since it was written
- `copy_structure` — the copy between `dict()` and `copy.deepcopy()`, rebuilding nested dicts and lists and passing every other value through. A shallow copy isolates one level, which is the gap a "returns a copy" docstring most often turns out to have; `deepcopy` closes it but overshoots, duplicating a connection pool or a lock and raising on anything unpicklable. Promoted from a private helper in `dataknobs-config`, which now imports it; the extraction surfaced a defect the private version's call sites never hit, since an `id()`-keyed memo can answer for an unrelated object once a source is freed
- `PluginFactory[T]`, the exported alias for the three shapes a `PluginRegistry` accepts under a key. The awaitable arm is new to the declaration — the class had accepted it at runtime and documented registering it since it grew `get_async`, so the exclusion reported as a type error against the documented pattern's own tests
- `Capability.VECTOR_PERSIST`, `VECTOR_INDEX_TUNING` and `VECTOR_DOCUMENT_API` with a `vector_storage` family, and `supports_capability(host, capability)` — the non-raising half of the guard pair, sharing one implementation with `require_capability` so the two cannot read a host differently. `host.supports(...)` raises `AttributeError` on an object that is not a contract host; `require_capability` already read that as "no", and a caller asking first had no way to get the same reading
- `dk_daemon_thread_names()`, and bridges register their thread names

#### Changed
- `PluginRegistry.get()` and `get_async()` no longer re-wrap a `NotFoundError` or `OperationError` raised by the factory itself, matching what `create()` and `create_async()` have done since the class was generalized. The wrapper exists because a factory's failure text is a driver's or an SDK's and can carry the connection URL it was handed; that reason does not reach our own error types, whose text we wrote and already bounded. A composite factory that misses in a sub-registry now arrives as the `NotFoundError` itself, so a caller's `except NotFoundError` catches it
- `lifecycle`'s teardown table scopes its blessing of `async def close()` to a collaborator whose shape the holder knows statically. It does not extend to one held in a heterogeneous registry, where the method's name is all the holder can route on and an `async def close()` is indistinguishable from a synchronous one until its coroutine has been discarded

#### Fixed
- `PluginRegistry`'s class-factory gate refused the implementation it exists to admit whenever `validate_type` was a Protocol carrying a property, raising `issubclass`'s own `TypeError` before any class was judged. Such a Protocol is now checked against its declared members — the same check `issubclass` performs for a method-only one, so a property-carrying Protocol and its method-only twin reach the same verdict on the same class. A Protocol never decorated `@runtime_checkable` is reported at registration naming the missing decorator, and both gates now share one implementation, so `set_default_factory` no longer rejects a class without naming it
- `validate_type` admitted a Protocol's async twin, because a `@runtime_checkable` check asks only whether a member is present. Both checks now compare each declared member's async-ness against the candidate's and refuse a mismatch either way, naming the registry, the key and the member; properties are skipped, having no async-ness to disagree about. Asking inside the gate rather than renaming the colliding members reaches a twin pair a *consumer* defines, and leaves `isinstance` answering `True` both ways — that answer belongs to the language rather than to this registry
- a synchronous `get()` or `create()` handed the caller an un-awaited coroutine when the factory was asynchronous, and `get()` cached it, so the first caller awaited it successfully and every later one received the same exhausted object and a bare `RuntimeError` naming neither the registry nor the key. The failure was inverted with respect to `validate_type` — a registry that gated raised cleanly and one that did not was silent, so the stricter configuration was the safer one by accident. Both now refuse an awaitable before anything is cached, naming the async method to call instead
- `get_async()` did not await every awaitable, testing with `asyncio.iscoroutine` where `create_async()` used `inspect.isawaitable`, and its `except Exception` re-wrapped the type guard's own message — alone among the four entry points in saying nothing about what was wrong. The four now share one invocation seam, parameterised by the two things that genuinely differ: the call arity, and whether the caller can await
- `run_callback` and `run_callback_off_loop` handed back an un-run async generator instead of refusing it, in all three spellings. `iscoroutinefunction` answers `False` for an async generator function so the synchronous arm merely *constructed* one without running a line of its body, and `isawaitable` answers `False` for the object, so the result-level net missed it too. Both now raise `TypeError` naming the shape: collecting a generator means choosing a policy that belongs to the surface which knows what the items are for
- `CallbackRegistry.fire` refuses an async callback that is an object. It asked `iscoroutinefunction`, so such a callback was classified as synchronous and *called* rather than refused, producing a coroutine dropped on the floor — no exception, no callback, no trace
- an unclosed `SyncLoopBridge` could not be reclaimed, reported, or collected. The loop thread's target was a bound method of the bridge, and `Thread` holds its target while the thread runs, so a bridge nobody closes was referenced by its own live thread — unreachable from any caller, alive in `threading._active`, holding its loop and that loop's self-pipe descriptors for the life of the process. None of it was observable, the thread being a daemon: a server building one bridge per tenant leaked a thread and two descriptors per tenant, silently. A bridge given the documented `thread_name=` was also invisible to the leaked-thread guard
- `DataknobsError` documented its two context arguments as merged, and they are not — the constructor resolves `details or context or {}`, so a consumer following the docstring and passing halves of one dict to each would silently lose one. Behaviour is unchanged; only the claim about it was wrong

### dataknobs-config [0.7.1]

#### Changed
- `EnvironmentConfig` copies its hand-outs through `dataknobs_common.copy_structure`, the private helper having become a public utility once a second package needed the same copy. Behaviour is unchanged except that the shared version keeps its memo's sources alive — a latent defect these call sites never triggered

### dataknobs-xization [2.2.1]

#### Changed
- the documentation filenames are lower-hyphen, matching the workspace's one-document-one-name rule — a shared name is what lets one page be served from both the package tree and the documentation site with its links correct in each. The links that pointed at the old spellings are updated, and the ingestion index now distinguishes the reference from the consumer-facing guide rather than describing both the same way. No source change and no consumer-visible change: a maintenance release, cut so the workspace carries one version set

### dataknobs-data [0.11.0]

#### Added
- `TextEmbedder` — the one shape for "turn text into vectors", batch-only, async-only, returning `list[list[float]]` and carrying `dimensions` and `model_id`. `dataknobs-llm` ships implementations; the protocol lives here because `data` cannot import `llm`
- `embedder=` on the vector write paths and on `VectorTextSynchronizer`, `VectorMigration`, `IncrementalVectorizer` and `DedupChecker` — the classes that embed a corpus over time, where a caller holding an embedder had to unwrap it into a callable and throw away the identity travelling with it. Passing one defaults `model_name` to the embedder's `model_id`: the two were independent parameters, so nothing stopped a caller naming one model while embedding with another, and the name is the staleness key. `VectorTextSynchronizer` and `IncrementalVectorizer` now raise at construction given neither source or both, rather than failing after a query, a batch and a partial write
- `SyncTextEmbedder` — how a synchronous site reaches an async embedder, owning one `SyncLoopBridge` so it is callable from plain sync code and from inside a running loop alike. The five synchronous signatures that already declare its shape are therefore unchanged; giving the protocol a synchronous twin so they could take one would restore the second shape it exists to remove
- `CachedEmbedder` and the narrow `VectorCache` port it takes, keyed on `(model_id, text)`; `DeterministicEmbedder` in `dataknobs_data.testing`, stable across processes with distinct texts landing near-orthogonal so a *ranking* can be asserted; `default_model_name`, `embed_texts` / `embed_text` / `require_embedding_source`, `call_embedding_fn_batch`, `BatchVectors`, and `coerce_operator` / `coerce_sort_order`
- `VectorStore` implements `CapabilityContract`, and `save()`, `load()` and `create_index()` are declared on it so a capability answer can be acted on without an `isinstance` downcast. New doc, `docs/vector-store-capabilities.md`, whose matrix is rebuilt from the classes by a test rather than maintained by hand
- `SyncVectorOperationsMixin` and `AsyncVectorOperationsMixin`, one per lane, with `VectorOperationsMixin` remaining as a name for the async one

#### Changed
- **Breaking:** `save()` and `load()` raise `CapabilityNotSupportedError` instead of returning quietly when a store has no `persist_path`, leaving the caller with nothing on disk to restore from and no indication why. The `hasattr(store, "save")` probe is the case to look for and it does not announce itself: it now answers `True` for every backend, so a guard written that way stops guarding. Use `store.supports(Capability.VECTOR_PERSIST)`
- **Breaking:** a `ClusterTopicIndex` that cannot run raises `StrategyUnavailable` rather than returning an empty topic list, which already means something else — the index ran and matched nothing — so a caller could not tell a misconfigured index from a vocabulary gap. The `TopicIndex` protocol was silent on the question, which is why its two implementations had each answered it independently
- **Breaking:** `SyncDatabase.config` and `AsyncDatabase.config` are read-only properties. A writeable attribute on one base against a read-only property on the other is a combination no class can have, so a type checker held every migrated backend impossible and stopped checking from the construction branch onward. Reading is unchanged; assigning now raises
- the vector-operations mixins take everything after `query_vector` by keyword in `vector_search`. The implementations did not agree on positional order, so a fourth positional argument already meant the metric on some backends and the filter on others; what changes is that a call written against the abstract declaration can no longer be written in the one form that was never portable
- an unrecognized operator or sort-order string raises `ValueError`, `vector_search` names its field parameter `vector_field` on every backend, `update_vector` reports whether the write landed, and `VectorizationResult` carries `skipped` so `processed` counts only what was written
- `BatchProcessor` runs a synchronous per-item callback on a worker thread, and `embedding_fn` is optional on `bulk_embed_and_store` where an `embedder` supplies the vectors

#### Deprecated
- `ConnectionPool` and `ConnectionPoolConfig` in `dataknobs_data.vector.optimizations`, in favour of `dataknobs_data.pooling.ConnectionPoolManager`. No backend ever used it and it is broken in ways not worth repairing in place: `acquire` awaits the caller's factory while holding a `threading.Lock`, so two coroutines acquiring from an empty pool deadlock the entire event loop — past the reach of any `asyncio.timeout`, since the loop thread is the thing blocked. Four of the five config fields are read nowhere and each now says so on itself; they stay, because removing one from a published dataclass breaks any caller passing it by keyword

#### Fixed
- fourteen dispatches called a consumer's callback without deciding whether the result needed awaiting. Three asked `iscoroutinefunction`, which reports a callable object as synchronous; the rest asked nothing at all, the commoner spelling and invisible to any check keyed on that predicate. Progress callbacks were dropped silently, `transform_fn` stored the coroutine as the migrated record, and `async_filter_stream`'s predicate answered `True` unconditionally because a coroutine is truthy. A guard now reads the AST of every `async def` here and fails on either shape, replacing a regex census that reported green over three unchecked dispatches in a file it was reading
- semantic dedup could not tell "unrelated" from "different vector space". `DedupChecker.register` held an embedder carrying a `model_id` and recorded none of it, so after a model swap `check()` scored old vectors against new ones and answered `unique` with an empty `similar_items`. It fails **open** — duplicates are admitted, and an admitted duplicate is silent by construction. Disagreement is now reported on `DedupResult.mismatched_model_ids` rather than corrected: the scores are still the only answer available, but a caller can tell they are meaningless
- two write paths stored vectors that named no model, permanently exempt from the model half of currency while four sibling writers recorded it. A digest answers whether the *text* changed and cannot answer whether the *model* did, since identical text through two models gives one digest and two incompatible vector spaces
- `VectorTextSynchronizer` decided currency on `model_version` alone, which the embedder path never sets, so `model_name` was written by two sites and compared by none and swapping embedders left every old vector reported current forever. A stored `None` is still not a mismatch, so no corpus re-embeds on upgrade
- `AsyncElasticsearchDatabase.bulk_embed_and_store` was a stub — a logged warning and `return []`, indistinguishable from "there were no records" — so a caller embedding a corpus got a successful-looking no-op and an empty index. Verified against a live cluster, including that the content digest survives the round trip
- `VectorStore.search_similar_records` ran a synchronous `fetch_records` on the event loop, stalling every other task for the whole fan-out, once per search. The returned value was correct either way, which is why nothing caught it
- `SyncTextEmbedder` blocks the caller's loop and now says so; `CachedEmbedder` no longer trusts the cache's batch length; a synchronous batch `embedding_fn` no longer runs on the event loop; and a plain `def` returning a coroutine now yields a vector rather than the coroutine
- `IncrementalVectorizer.run_batch` can return, terminates on a record it declines to write, and does not count a failure as a win. `wait_for_completion` waits for the work: it polled `qsize()` and returned as soon as that was zero, which two states satisfy and neither means done — nothing enqueued *yet*, since `start()` creates the loader as a task, and everything in flight, since `qsize()` drops when a worker takes a record rather than when it has written it. Every wait in the class now races the work against shutdown instead of polling for it
- `ChromaVectorStore.search_documents` accepts `include_timestamps`, which its two siblings gained and it did not — it is Chroma's alone, so it appears in no cross-backend parity fixture, which is how it kept the older signature while the methods around it moved
- a sync backend's inherited vector methods run, `Operator.NOT_LIKE` is reachable from `Query.filter`, a `ComplexQuery` is answered by the file and S3 backends, a record read back carries the id it was read under, a record with no fields can be read back, and a JSON file whose top level is not a mapping no longer reads as an empty store
- the other four copies of the text assembly — `VectorSyncMixin`, `BulkEmbedMixin`, `AsyncBulkEmbedMixin` and `IncrementalVectorizer` each rebuilt the embedder's input independently, three of them on a hardcoded separator, and the first three recorded no digest, without which the fields they write are unjudgeable by anything and count as current forever. `VectorSyncMixin.sync_vectors_with_text` additionally decided staleness by comparing the *set of source fields* and nothing else, so an edit to the text was invisible to it
- the Elasticsearch and flat-file paths that dropped a vector field's metadata, and `ChangeTracker` and `VectorTextSynchronizer` no longer disagreeing about what changed

#### Removed
- **Breaking:** the `EmbedFn`, `BatchEmbedFn` and `ClusterEmbedFn` aliases in `dataknobs_data.sources`, replaced by `TextEmbedder` in one change with no deprecation cycle. They were three exported names for two types, and `EmbedFn` meant the batch shape imported from `sources` and the single-text shape by the same spelling inside `cluster_index.py`, so a consumer could not tell which they had from the name. With them go two parameter names: `embed_fn` is `embedder`, and `ClusterTopicIndex.build` takes a single `embedder` where it took a batch and a single-text callable that nothing stopped a caller pointing at different models
- the `{vector_field}_content_hash` record-field comparison. Nothing in the library ever wrote that sibling field, so in production it compared `None` against a fresh digest and called every record stale; the only writers were two test fixtures, which is what made a dead branch look covered

### dataknobs-llm [0.9.0]

#### Added
- `LLMProviderEmbedder` and `create_text_embedder` — an embedding provider presented as the `TextEmbedder` seam `dataknobs-data` declares, satisfying it structurally and importing it under `TYPE_CHECKING` alone, so nothing pulls `dataknobs_data.vector`, and numpy behind it, into an `llm` import. There is no conversion in it, and that absence is the seam's justification: what the adapter adds is the two things a bare provider cannot answer in the shape a *stored* vector needs — a settled `dimensions` and a stable `model_id` of `provider:model`. A declared width is checked against the first batch rather than trusted. No new config type: an embedder config *is* an `LLMConfig`

#### Changed
- `list_conversations` and `search_conversations` refuse a `sort_order` they do not recognise, and the `WizardStateSnapshot` deprecation names its versions

#### Fixed
- a stated embedding width was honoured by one provider of five and accepted-and-discarded by the rest, so a config asking `text-embedding-3-large` for 512-dimensional vectors received 3072 — valid vectors, six times wider than requested, at six times the storage and the price, with nothing raising at any layer. The first component to object was a vector store rejecting the write, and that message names the store rather than the misconfiguration. There is now one rule resolved on the base: the per-call keyword beats the config, and a width is forwarded only to a model that advertises accepting one
- the FSM `LLMResource` no longer invents an embedding it could not compute. Its sync `embed()` routed to one of three per-provider methods and answered everything else with a constant vector of a plausible width — which a vector store accepts, indexes, and returns as a nearest neighbour to every query, with no component downstream positioned to notice. It also builds its provider once instead of per call, and uses the configured credentials and endpoint
- the embedding cache served a vector of the wrong width on a hit, and `CAPABILITY_ORDER` silently dropped any `ModelCapability` missing from it
- registering a `ModelMetadataSource` *class* in `model_metadata_sources` raised instead of registering it — the registry's own documented extension point, met on the obvious shape. Only the callable-factory route worked, and it is the only one this package's tests had exercised. Fixed in `dataknobs-common`'s `PluginRegistry`
- `AsyncLLMProvider.stream_complete` is declared as what it returns, `VectorRetriever.index_documents` and `.retrieve` run, and `LLMWorkflow` builds an FSM again with a terminal end state

### dataknobs-bots [0.13.0]

#### Added
- wizard arcs carry a name, and a transition record says which one fired. A stage may declare several transitions to the same target, and the compiled arcs were anonymous, so both siblings reported the same `"<source>-><target>"` string. The derived form extends the FSM's own fallback rather than replacing it, so a reader who knows the old string still reads the prefix and only the discriminator is new; it is persisted on `TransitionRecord.transition_name` and carried into `ExecutionRecord`, where the field has always existed with nothing to put in it. `DynaBot.get_wizard_transitions(conversation_id)` returns the records
- `resolve_middleware_class` — validate a middleware spec without running its constructor. `resolve_middleware_from_spec` is now that function plus one `cls(**params)`, so a caller that checks with one and installs with the other applies the same rules at the same points. The check is inherently weaker than the build: a constructor that rejects its params is undetectable without running it, so a clean answer means *this spec is installable*, not *this bot will start*. Resolving a dotted path still imports its module, so the trusted-configuration rule governs this with more force rather than less
- `SourceFactory` is `PluginFactory[GroundedSource]` rather than a hand-written copy of the registry's contract, which omitted the asynchronous arm — so this module's own asynchronous registration was the one call its own declaration refused
- `WizardStateSnapshot` carries the current stage's configuration, and `BotTestHarness.get_transitions()` returns the same records for the harness

#### Changed
- a dead topic index is no longer reported as a corpus with nothing in it. `HeadingTreeIndex` returned `[]` in lazy mode with no `vector_query_fn` and swallowed everything its vector query raised, and the retrieval loop reads an empty topic index as a vocabulary gap — so a wiring fault silently rerouted the turn and logged the corpus as the cause. The results do not change, since `GroundedReasoning` takes the same fallback; the record does, to a WARNING naming the index and what it lacks
- the `cluster` topic index is built with a `TextEmbedder`, a small adapter presenting the knowledge base's per-text `embed` as the batch shape rather than the index accommodating both arities. The calls stay sequential — a knowledge base's `embed` is usually a provider call behind a shared client, and firing a whole corpus at it concurrently is a rate-limit decision this adapter has no standing to make — so a cluster index behaves as before. Requires the `dataknobs-data` release above, since the parameter it replaces is gone there with no deprecation cycle
- `DynaBotConfigSchema.validate` applies the `$resource` marker rule to the whole config file rather than its `bot:` section. The rule is a property of the config format rather than of any schema, and the component loop beside it visits registered names only, so a reference block under any other key was one nothing looked at

#### Fixed
- the wizard no longer reports the wrong arc when two transitions share a target, and `TransitionRecord.from_dict` no longer raises on a key it does not recognise
- a snapshot's payloads are copied in depth, so writing through one cannot reach the wizard's own state; the same for its `suggestions`, roadmap and recorded transition data, and `DynaBot.get_wizard_state()` returns a dict the caller owns
- a subflow arc's target was derived in two places, and stage metadata now records the compiled target beside the condition

### dataknobs-fsm [0.5.0]

#### Added
- `AsyncClosable` and `AsyncCleanable`, `ResourceManager.unclosed_providers` — a mapping of provider name to the teardown failure that stranded it — and `SimpleFSM.get_state(name)`, the singular accessor its async sibling already had. One resource surface across `SimpleFSM`, `AdvancedFSM` and `AsyncSimpleFSM`

#### Changed
- one judgement about whether a callable is async, made in one place. Nine modules asked it for themselves in five spellings — bare `iscoroutinefunction`, that plus a check on `__call__`, a check on `type(f).__call__`, an `_is_async` attribute hint, and no check at all — and all now route through `dataknobs-common`. The repository-level adoption guard covers `packages/fsm/src` as a result, so a new dispatch that skips the judgement fails a test rather than reaching a consumer
- **Breaking:** `register_provider` on a closed or closing `ResourceManager` raises where it previously returned. The provider it accepted was never torn down, so the call had no effect a caller could rely on beyond leaking the transport
- a configuration key the schema does not declare now fails the load; `$include` and `$import` are stripped with a warning when the loader is not permitted to follow them; `StateSchema` is constructed from a JSON Schema mapping; and the database streaming classes take a sync database and say so
- `IORouter.add_route` and `IOBuffer` accept async callbacks in their signatures, and `AsyncStreamContext.stream_async` and `StreamingFileProcessor` accept them too

#### Fixed
- the two halves of `ResourceManager` teardown disagreed about the registry they share, and a provider could be lost between them. `close()` iterates under the lock; `cleanup()` read the same dict unlocked and suspends three times holding nothing — so a provider registered mid-teardown arrived after the classification sweep, was never closed, was never recorded as a failure, and was removed by the clear that ends `cleanup()`, leaving the transport open with nothing naming it. A concurrent `unregister_provider` could abort the sweep outright. It now classifies a snapshot taken under the lock, rather than a wider critical section the re-entrant lock could not make safer
- a provider whose teardown had to be awaited was torn down as though it did not. Teardown routes on the method's *name*, the only thing a registry of unrelated provider types can route on, and nothing held a provider to it: the contract declared no teardown method, and the base gives every provider a `close()`, so probing distinguished nothing. `AsyncDatabaseResourceAdapter` did not override `close()`, so a synchronous teardown released the handle list and never touched the database — no coroutine is created there, so nothing warned. `close()` also abandoned teardown at the first failure, and `cleanup()` could not name it
- async callbacks that were silently doing nothing now run: an arc condition, sink, transform, validator, compensation action or wrapped operation whose coroutine was previously discarded is now awaited. An arc condition written as a callable object matched every record, an async record predicate admitted every record, `IORouter` used its route's `condition` and `transform` without awaiting them, `IOBuffer` lost its overflow when the handler was a callable object, and the resilience wrappers recorded outcomes for calls they never made
- `ParallelIOExecutor` silently skipped every synchronous provider and ignored `max_workers`, which `__init__` stored and nothing read; a provider list holding both an async and a synchronous provider for one destination therefore now receives **two** writes where it previously received one, and `write_all` returns `None`, so nothing surfaces that
- `IOBuffer` discarded the overflow when the handler failed, and `IOBuffer(max_size=1)` never drained and grew without bound
- `AsyncStreamExecutor` credited records to a sink that never received them, `AsyncStreamContext.stream_async` reported success over writes that never happened, and both it and `AsyncBatchExecutor` stopped reporting progress partway
- `StateDefinition.validate_data()` raised `TypeError` on every FSM built from configuration, `AsyncSimpleFSM.validate()` raised on an FSM with no start state and was documented as returning `bool`, and `StreamExecutor.create_multi_stage_pipeline` built a pipeline with no stages
- `RecoveryStrategy.DEADLINE` could not run a synchronous function, and `FunctionWrapper` misread a `functools.partial` around an async callable
- the data-mode documentation taught a configuration shape that is silently ignored, the Advanced API's `DataHandler` example could not be instantiated, and a state `schema` was documented as validating data on arrival

#### Removed
- **Breaking:** `AdvancedFSM.set_data_handler()`. It assigned `self._engine.data_handler`, a name the execution engine neither declares nor reads, so every handler passed to it was silently ignored — the call returned `None` and changed nothing. Calling it now raises `AttributeError`, which is the first diagnostic the method has ever given. Data handling is selected by **mode**: `data_mode.default` at the top level of a configuration, overridden per state by that state's own `data_mode`

### dataknobs-legacy [0.2.2]

#### Changed
- the pinned sibling versions move to the ones released today — `dataknobs-common` to 3.2.0 and `dataknobs-xization` to 2.2.1, with `dataknobs-structures` and `dataknobs-utils` unchanged. This package pins with `==` rather than `>=`, so its version tracks its siblings' whether or not its own code moves, and no source changed here beyond `__version__`. The pin is not cosmetic this time: `dataknobs-data` 0.11.0 requires `dataknobs-common>=3.2.0`, so a v0.2.1 still asking for `dataknobs-common==3.1.0` cannot be resolved in an environment that also holds a package from this release

## Release - 2026-08-26

Three changes run across the workspace in this release.

**A config key nothing recognises is an error rather than a silent default.**
`dataknobs-common` gives `StructuredConfig` an unknown-key policy —
`_UNKNOWN_KEYS`, `_INPUT_KEYS` and `accepts()` — defaulting to the lenient
reading so no existing config class changes, and `dataknobs-data` opts in once
on `DatabaseConfig`, so all fourteen of its backends — seven sync, seven async
— inherit it. The default was wrong specifically for that family: every
connection field has a working default, so a config built from misspelled keys
did not fail, it succeeded against the wrong store —
`create(backend="postgres", hosst="db.internal")` connected to `localhost` and
logged nothing. Turning the key into an error is what made the documentation
readable as code, and it turned out to be wrong in sixteen places: pool sizes
under SQLAlchemy's spelling, `hosts` passed to the sync Elasticsearch backend,
`username`/`password` where the field is `basic_auth`, `connection` for
`connection_string`, and two file-backend options that have never existed. The
same sweep reached `dataknobs-llm`'s conversation-storage examples, a memory
bank in `dataknobs-bots` sending a table name to backends that have no table,
and `dataknobs-fsm`'s `InMemoryStorage`, which set two connection parameters
the memory backend has no fields for and documented itself as applying them.
Errors name the offending key, suggest the nearest accepted spelling, and list
the accepted set; routing keys (`backend`, `factory`, `name`, `type`) still
pass through.

**A component that cannot do its job no longer reports success.** The shape
recurs in every package and is the same each time: a failure absorbed one frame
below the guard that would have reported it, arriving at the caller as an
ordinary empty answer. A `DatabaseSource` whose database was misconfigured
returned "no matching records" on every call, indefinitely, while reading as
healthy; a `ClusterTopicIndex` that could not embed returned an empty topic
list, which grounded retrieval reads as a vocabulary gap, so a broken embedder
silently rerouted the turn to a different strategy and named the wrong cause.
Both now raise, and the retrieval loop's existing per-source guard — which had
no test and was unreachable through either — drops the source and logs what
actually happened. In `dataknobs-bots` a wizard tool wrote into a throwaway
dict and reported success, a subflow that failed discarded the work and
reported success, `preview_config` rendered a config `build()` would refuse and
said nothing about it, and `skip_default` replaced a value the user had set
five turns ago with no log line. In `dataknobs-common`, `safe_eval` decided
whether to prepend `return` with a substring test, so `return_code == 0`
evaluated `False` forever with `success=True` and `error=None` — a wizard
transition guarded by one never fired, and nothing logged anything.

**The `$resource` marker rule is a callable, applied at every depth.**
`dataknobs-config` exports `collect_marker_violations()`, which walks any
config tree and reports each breach without needing an environment and without
raising — the rule, where only `RESOURCE_MARKER_KEYS`, the vocabulary, had been
exported before. Offering the set alone is what left every caller to write its
own rule around it, and `dataknobs-bots` had done exactly that: a transcription
of one clause, applied to one mapping, which agreed with the resolver about a
reference handed to it directly and disagreed about everything else — a nested
reference, a misspelled `$resource` selector, and any section no schema is
registered for. The transcription is gone and both packages now share one
definition, so a marker defect is described one way whether it surfaces as a
lint or as a failed build. More configs are reported invalid than before;
everything newly reported already failed at resolution, in whichever deployment
lacked the resource, instead of at config-lint time. `dataknobs-config` also
now reports a config that contains itself — a YAML anchor builds one directly,
and both readers of the format descended it until the stack ran out.

### dataknobs-common [3.1.0]

#### Added
- an unknown-key policy on `StructuredConfig`: `_UNKNOWN_KEYS` (`"ignore"` by default, so no existing class changes), `_INPUT_KEYS` for input spellings `_normalize_dict` consumes that are not themselves fields, and `accepts(key)` — the question a caller composing a config for a statically-unknown target has to be able to ask. Under `"raise"` an unrecognised key names itself, offers the nearest accepted spelling, and lists the accepted set
- `safe_eval_validate(expression, *, restrict_builtins=True)` — the static pass `safe_eval` runs before evaluating anything, as a callable. `safe_eval` now calls it, so there is one implementation and the two answers cannot drift. A refusal and a runtime failure both arrive as `success=False` with an unstructured `error`, so a caller holding only an `ExpressionResult` cannot tell "this will never run" from "not satisfied yet" — which for a config-authored wizard condition means the stage never fires and nothing reports why. It never raises, and `None` is not a safety review
- class-definition validation for the class-level policy attributes. `_UNKNOWN_KEYS = "Raise"` compared unequal to `"raise"` and selected the lenient policy in a class that read as opted in to the strict one; `str` is iterable, so `_INPUT_KEYS = "connection_string"` unioned in ten single characters and still rejected the alias it was written to declare, and `_SENSITIVE_FIELDS` failed the same way while turning a field-name test into a substring match; omitting `ClassVar` makes the dataclass decorator treat any of them as a field. Checked at runtime rather than left to the type checker, because the subclasses that matter are consumers' and a library cannot assume its consumers run mypy

#### Fixed
- `safe_eval`'s `return` prefix is a token test rather than `startswith("return")`, so `returned_value > 1` and `return_code == 0` are evaluated as expressions instead of being left unwrapped, returning `None`, and coercing to `False` with `success=True` and `error=None`

#### Changed
- `safe_eval` and `safe_eval_value` accept any `Mapping` as `scope`, so a caller holding a read-only mapping no longer has to copy it to satisfy the annotation

### dataknobs-config [0.7.0]

#### Added
- `collect_marker_violations()` and `MarkerViolation` — the `$resource` marker rule as a callable, returning a dotted path and the resolver's own sentence per breach, for a validator or config-authoring tool that has no environment and no permission to raise. It descends into a reference's inline defaults unconditionally where resolution walks only those an environment does not override, deliberately: a malformed reference inside an overridden default goes live the day that override is removed, and a validator's subject is the authored config rather than one deployment of it

#### Fixed
- a config that contains itself is reported rather than followed round. Both readers of the `$resource` format descended a self-containing tree — which `yaml.safe_load` accepts without complaint — until the stack ran out. The existing cycle guard could not reach it: it tracks resource *identities*, and a block reaching itself is a cycle in object identity with no `$resource` key involved. Both guards now live on one object threaded through the recursion, and it is a stack rather than a visited set, so an anchor reused for its ordinary purpose still resolves

#### Documented
- the configuration guides teach `StructuredConfigConsumer` rather than `ConfigurableBase`. The deprecated base is soft-deprecated by design and raises no runtime warning, so documentation was the only channel through which a new adopter could learn it is going away — and the guides were the channel recommending it

### dataknobs-structures [1.0.17]

#### Changed
- this package's tests joined the linted set at a ceiling of zero, alongside the sources that graduated in v1.0.16. No source change and no consumer-visible change; released here so the workspace carries one version set

### dataknobs-utils [2.0.2]

#### Fixed
- an emoji immediately after the Status Counts block was dropped from the loaded data. `EmojiData._load_emoji_test` verified the file's status tallies with a nested `for line in f` over the *same* handle as the outer loop, so the line that terminated the inner loop had already been consumed when the outer loop resumed and was never classified. In the shipped Unicode 15.0 data that line is blank and nothing was lost, which is what made it worth fixing rather than leaving: whether an emoji went missing depended on a property of the input file that nothing stated and nothing checked

### dataknobs-xization [2.2.0]

#### Changed
- `format_heading_for_display` raises `ValueError` when `headings` and `heading_levels` differ in length instead of silently dropping the excess. The function is public and both lists come from the caller, so a mismatch was a caller error that produced quietly truncated output. Callers passing equal-length lists — every caller inside this package — are unaffected. The same tightening covers the two internal heading walks, where the two lists are built together and a mismatch would mean a construction bug rather than bad input

### dataknobs-data [0.10.0]

#### Changed
- **Breaking:** every database backend config rejects a key it does not accept, rather than discarding it. Set once on `DatabaseConfig`, so all fourteen backends inherit it and so does a backend added later. The "synthesized default values" warning could not cover this — it fires when *recognized* explicit keys mix with defaults, and an unrecognized key enters neither bucket, so the config read as "nothing was configured" and the case most in need of the warning was the one it structurally could not see. A call that now raises was already not doing what it read as doing; ask `CONFIG_CLS.accepts(key)` to supply a key only some backends have
- the backend registry's `config_options` metadata named keys the config classes reject. It is the programmatic equivalent of a documented sample — read by a consumer building a config form — and carried all three defects found in the markdown: a field belonging to the sibling backend, another library's vocabulary, and a field that never existed

#### Fixed
- **Breaking:** a `DatabaseSource` raises rather than reporting a store it cannot reach as a store with nothing in it. Both of its searches were wrapped and logged, so a misconfigured database answered "no matching records" on every call while reading as healthy. The wrapping added no resilience the caller did not already have — the grounded retrieval loop guards each source, logs one that raises, and drops it for that turn — and absorbing the failure one frame below meant that guard never fired. A partly-failed search now raises rather than returning the part that worked
- **Breaking:** a `ClusterTopicIndex` raises rather than reporting an index it cannot run as one that found no topics. It landed harder than its sibling because an empty topic index is read as a vocabulary gap, so the turn fell back to plain text retrieval and named the wrong cause. Embedding seed chunks keeps its per-chunk tolerance, now reported at WARNING rather than DEBUG, but a pool where every chunk failed raises
- `SensorDataGenerator` draws from its own stream, so a seed means what it says. It seeded the module global and then drew from it across every later call, making its output a property of the whole process: anything else drawing in between shifted the sequence, and constructing one silently reseeded every other consumer. Measured across 3,000 constructions with an unrelated draw interleaved, six distinct outcomes before and one after. `seed=0` also now seeds — the guard was `if seed:`, which read zero as "no seed given"
- sixteen documented factory calls named a field no backend has, invisible while unknown keys were dropped: the sample ran, and what it configured was the default. The two forms most consumers copy broke rather than merely misconfigured — a bot's `conversation_storage:` block reaches the factory with every key it carries, so the documented production Postgres sample raised at startup for anyone who copied it. A new test checks every documented factory call, backend constructor and YAML block against the real config class via `accepts()`
- `Not` is exported from `dataknobs_data.validation`. Of the three logical combinators the only one that negates was the only one unreachable by the path the other two use; a test now derives the expected export set from the constraints module

### dataknobs-llm [0.8.0]

#### Added
- `ToolExecutionContext.wizard_data()` — a supported way for a tool to reach wizard data. The only route before was guarded by a check the accessors around it consistently collapsed into an empty dict, so a tool run outside a wizard appended to a fresh throwaway, saw its own write, and reported success. The new accessor returns `None` there, deliberately, so the condition is one a tool can detect and report
- `ConversationState.live_wizard_state` — a per-turn channel a reasoning strategy publishes live wizard state on, preferred by `ToolExecutionContext.from_manager` over rebuilding from persisted metadata. It sits on `ConversationState` rather than inside `metadata` because wizard data is deep-copied on restore precisely so live state and persisted metadata cannot share a reference

#### Deprecated
- `WizardStateSnapshot` is now `ToolWizardState`; `dataknobs_bots` exports an unrelated and much larger dataclass under the same name. The old name remains an alias in `dataknobs_llm.tools` and `dataknobs_llm.tools.context` until 1.0.0, when it is removed, and warns when read from either, while type checkers still resolve it to the class

#### Fixed
- extra arguments to `LLMProviderFactory.create()` reach the provider. The signature has taken `**kwargs` and the docstring has described them as passed to the constructor since before the provider registry existed, and neither branch ever passed them on
- `create_llm_provider()` returns the one provider the call can produce, being overloaded on `is_async`; `LLMProviderFactory.create()` keeps returning the union, because `is_async` is a constructor flag there and the method must stay callable through the `Config` factory protocol
- the factory's sync arm names the class it actually returns. It declared `AsyncLLMProvider | SyncLLMProvider`, but `SyncProviderAdapter` wraps an async provider rather than subclassing `LLMProvider` and no `SyncLLMProvider` subclass exists in tree, so that arm was uninhabited and held down by a `# type: ignore`
- the provider registry produces providers, not provider classes. `PluginRegistry[T]`'s parameter is what a registration *produces*, and the union this leaked had made `provider.complete(...)` statically ambiguous between a coroutine and an `LLMResponse` wherever a sync provider was requested
- seven documented conversation-storage examples named database config keys the backends do not have; the field is `path` in every case. Each built a database at the config default rather than where it named — the SQLite ones at `:memory:`, so an example about persisting conversations to a file persisted nothing

### dataknobs-bots [0.12.0]

#### Changed
- **Breaking:** the five reasoning-strategy configs are keyword-only. Inheriting `greeting_template` from the new `ReasoningConfig` base forces it: a base field is declared ahead of every subclass field, so a defaulted one would sit in front of the wizard's required `wizard_config`. Four of the five would have raised on a call written against the old order; the wizard would not, because its second positional was `greeting_template` and became `config_base_path`, both `str | None` — so `WizardReasoningConfig(cfg, "Hello!")` would have constructed, with no greeting and a nonsense base path. Migration is mechanical: name every argument
- `greeting_template` is declared once for the reasoning-strategy family and read from one place. `ReasoningStrategy` has always documented it as universal, but each of the five configs declared it and each of the five strategies copied it onto itself, and a strategy that skipped either half was not reported — a config class that omits a key drops it rather than rejecting it. `ReasoningConfig` is exported for a consumer's own strategy config
- wizard bots honour the strategy-level `greeting_template` instead of discarding it. A documented limitation lifted rather than a bug fixed: the old behaviour was stated in the configuration guide, and nothing can have depended on it, but a bot that sets the field today will start greeting with it
- `preview_config` reports whether the config it renders is valid, returning `valid`, `errors` and `warnings` beside the render in all three formats and using the same keys `validate_config` returns. It reports **and** renders — a config with errors is still the thing being built. Consumers reading the output should expect the three new keys; the two paths with no config to render still return `{"error": ...}` and carry no verdict
- config validation enforces the `$resource` marker rule at every depth, so more configs are reported invalid than before. `marker_violations_result()` is exported for a consumer composing a validator pipeline of its own
- a `ConfigValidator` passed to `ValidateConfigTool` is additional rather than a replacement, so a consumer supplying both gets strictly more errors and never fewer. The failure directions are not symmetric: an extra error stops an author, a missing one misleads them. `ValidationResult.merge_unique` is added for composing validators that cover overlapping ground; `merge` still concatenates
- a `WizardFSM` stage accessor returns the type it declares. Stage metadata is authored config carried through uncoerced, so `can_skip: "no"` gave a truthy string from a method declared `-> bool` and `tools:` written as a bare string iterated character by character. The seven typed accessors share one read that substitutes the documented default and warns once per stage and field
- `WizardFSM` no longer reports a matched subflow transition as "none matched" — such a transition compiles to a self-loop arc, so a match leaves the FSM where it started and was indistinguishable in the log from a failed condition. A declined subflow push now logs the guard's decision on both branches, naming the conditions asked; a decline previously left no trace at all
- a condition that fails on a turn's data is logged at DEBUG rather than WARNING. `data['name']` before `name` has been captured is the ordinary state of a guard whose input has not arrived, and warning on it every turn for a correct config is how a log teaches its reader to skip it. WARNING is reserved for a condition the engine refuses
- loading a config with an ill-typed text field no longer raises: two of the loader's warning heuristics searched an authored value with a regex directly, so a non-string prompt or condition took the whole load down from inside a check that exists only to advise
- `WizardConfigBuilder.stage()` no longer declares a `skip_extraction` keyword, which is a per-turn state flag rather than a stage field; a registry-sync test now asserts every explicit `stage()` keyword names a field the loader reads
- the three places a bot builds an LLM provider from config call `create_llm_provider()` rather than the factory constructor, so the site can say which provider comes back

#### Added
- `greeting_template`, a wizard stage field for an opening line the stage says once — on the turn the stage first speaks, then stepped over, whatever the stage's mode. A structured stage had no way to open with fixed text: its `response_template` is deliberately re-rendered every turn, and the only escape was `mode: conversation`, which turns extraction off. Greeting a stage does not consume the render its `confirm_first_render` is waiting for
- the loader reports two config surfaces that parse, validate and read as correct while doing nothing: a `response_template` a `greeting_template` has made unreachable, a pushed subflow's `settings:` block, and `auto_advance: true` on an end stage. Warnings rather than refusals, which is this validator's contract for all eight of its checks — a subflow's `settings:` is not wrong, it is unread
- wizard transition conditions are checked when the wizard is loaded, so a condition the expression engine will refuse is named once with its stage and target. Nothing said so before evaluation, and by then a refusal is indistinguishable from a condition merely unsatisfied, because every wizard condition site passes `default=False` — load time is the last moment the report reaches the author
- `DynaBotConfigBuilder.build_unvalidated()` — the public name for building without validating, for callers that report a `ValidationResult` rather than raise on one; the config-toolkit tools had all been reaching through a private method from outside the class. Pair it with `validate()`
- `dataknobs_bots.config.injected_dependency()` and `InjectedCallable` — the one line a `from_config` needs to tell a live dependency from config data. Public because a consumer writing a tool with a `requires` entry has the same problem and no way to discover the answer otherwise
- `ContextAwareTool.missing_arguments()` and `missing_arguments_result()`, both overridable; `WizardState.replace_data()`; and `BotTestHarness.create(custom_functions=...)` with `WizardConfigBuilder.transition(derive=...)`

#### Fixed
- a subflow's `is_end` stage renders its `response_template`. The stage was entered and left inside one turn, so the pop ran in the same step and the parent's return stage rendered instead. The cost is not the usual missing line: a subflow that can fail ends on a stage whose whole job is to say *nothing was saved, and here is why*, and that refusal was the one message that never appeared — the flow discarded the work and reported success. The template renders before the pop, against the subflow's own data, and a failed render no longer takes the departure with it
- stage-dependent state resolves against the FSM that owns the stage. `WizardNavigator` holds both the main FSM and the subflow manager and each of its methods picked one by hand; five picked the main FSM, which inside a push does not have the current stage — so a subflow stage declaring `can_skip: true` was told it was required, its own navigation keywords were never found, back landed on the right stage and rendered one with no prompt, and a custom context template told the model a skippable stage was required. Every site now asks `SubflowManager.fsm_for_state()`. **Note:** stage-level keywords replace wizard-level ones per command, as they always have outside a subflow
- the read-only state snapshot describes the subflow stage it is standing on. A skippable subflow stage reported `can_skip: False` and empty `suggestions`, so a skip button disappeared and quick replies vanished for as long as the subflow was open, while `stage_index` reported `0` — a progress bar that jumps to the start whenever a subflow opens, contradicting the same object's `stages` roadmap. The mixed frame is now documented on the type
- `snapshot_from_metadata()` reports the same state its instance-method sibling does. `can_skip`, `can_go_back` and `suggestions` were never passed to the constructor at all, so they took the dataclass defaults in every flow — a UI on this path never showed a skip button and never showed a quick reply, subflow or not. `data` and `history` are now copied out of `fsm_state`, as the instance method already copied them; returned by reference, a consumer appending to `snapshot.history` on a type documented read-only silently rewrote persisted state
- restart inside a subflow leaves the subflow. `restart_cleanup` left the stack loaded, so the wizard reported the main flow's start stage while rendering the subflow stage's prompt and could not recover — restart, the escape hatch of last resort, was what wedged the wizard. It now unwinds through `SubflowManager.unwind_all()`, recording a `subflow_pop` per frame so the transition trail no longer holds a push nothing closes. Task completion and `transient` also survived the reset and no longer do
- amendments resolve a section against the whole flow and unwind to reach it. Membership is a property of the config rather than of where the user stands; acting on the answer does read the stack, so an amendment whose target is in the main flow while a subflow is open unwinds first, and one naming a stage in some other subflow is declined and logged
- a subflow guard reads what its own stage prepared. The condition on a `_subflow` transition was evaluated before the stage's pre-transition preparation ran, so a guard reading a key written by `routing_transforms:` or `derive:` fired a turn late, against a message the user wrote in answer to a prompt they never saw
- `advance()` can push a subflow. The non-conversational API never asked whether one should be pushed while still reaching `should_pop`, so it could be carried out of a subflow it had no way to enter
- a wizard tool's writes to collected data are no longer discarded, its reads are no longer a turn behind, and a tool on the first turn of a wizard that does not greet gets real state. `ToolExecutionContext` rebuilt wizard state from persisted metadata, which the wizard rewrites when the turn is saved — so a tool wrote where nothing would read it again and reported success, and where a stage declared `tool_result_mapping` the tool was *called* with this turn's values and *read* the previous turn's. `WizardReasoning` now publishes live state for the duration of the turn
- a flow change mid-turn no longer strands a tool on abandoned data: `WizardState.data` was rebound on a subflow push, pop and restart, and the three sites now refill the dict in place
- the KB tools report missing wizard state instead of writing into nothing, reaching wizard data through the public accessor
- `skip_default` no longer has to overwrite a value the user set. It was applied with a bare `dict.update`, so a key the user set five turns ago was replaced as readily as one never touched, with nothing left to say the value had ever been different. A stage declares `skip_default_mode: fill`, or a key states its own mode; `overwrite` remains the default. **One config shape changes meaning:** a key whose value names exactly `value` and `mode` is now read as an annotation
- a `navigation:` block is type-checked before use, wherever it was written. `enabled: "false"` is a truthy string, so a command the author turned off stayed on; quietest and worst, `keywords: "done"` was iterated into four one-letter keywords, so a user answering `d` triggered a command meant for `done` — nothing raised, nothing logged, and the config read correctly. Both readers now share one implementation, and four things authored as a keyword list share one predicate
- a stage field left unset is no longer reported as ill-typed; an absent field reaches the shared read as `None`, which is the registry's marker for "not declared" rather than a wrong type
- `validate_config` and `save_config` reach one verdict. The disagreement was symmetrical and only half of it was the tool's: `SaveConfigTool(portable=False)` — the constructor and `from_config` default — built through the unvalidated path and wrote to disk exactly the config `validate_config` had just refused. Both settings of the flag now validate; `portable` selects the output shape, not whether the config is checked
- every `ContextAwareTool` answers an omitted required argument rather than raising `TypeError` from the call itself. Nine tools carried that shape, which was also the `[override]` incompatibility the type checker reported at all nine sites
- a tool handed the dependency it declares now uses it. Five of the six built-in tools with a `requires` entry read only the YAML spelling of the key, so three discarded the live object and rebuilt their own while two put an already-resolved callable through `resolve_callable` and raised. Only `KnowledgeSearchTool` worked, and only because it defines no `from_config`
- a `database` grounded source works against a real store. It forwarded a `connection` string no backend accepts under any spelling, dropping every key that names a store, then never connected what it built — and a backend that needs connecting raises on every query, which `DatabaseSource` reported as an empty result set. **Breaking** for a config carrying an option no backend accepts and that was previously discarded
- the successors named by a deprecation are importable from the same place as what they replace: `dataknobs_bots.api` exported four deprecated names and none of the four registry names defined in the same module, so a consumer who read the warning and changed the name got an `ImportError` — the only working code was the deprecated code
- a `clarification_template` set without a `response_template` renders, and a conversation-mode stage renders it when the turn is streamed rather than only when buffered; the template-selection rule existed in three copies and now exists once. A stage the wizard auto-advances past no longer repeats an opening line it has already delivered
- a `storage_class` with no `create` names itself, raising `ConfigurationError` carrying the config key and the dotted path instead of a bare `AttributeError`
- the wizard loader no longer prepends `return` to a condition before handing it to the expression engine, which has done this itself since it was introduced; the loader's copy predated it and carried the same substring-test defect. The expression logged alongside a failed condition is now the one the author wrote
- an abandoned stream no longer leaks a turn's `turn_data` into the next one — the cleanup ran in turn finalization, which `stream_chat()` deliberately skips when the stream was not fully consumed, and now runs in the `finally` every turn driver executes
- `ErrorRaisingStrategy` accepts `greeting_template`, so building it from config no longer raises; a registry-driven test now holds every registered strategy to the universal contract, including one a consumer registers
- `WizardConfigBuilder.add_subflow_network()` produces a subflow the loader can read. Neither direction worked: `to_dict()` emitted a shape `load_from_dict` refuses, and `from_dict()` iterated a documented `subflows:` section as though it were a list. The method had no caller and no test anywhere in the tree
- a repeated wizard-validation message is emitted once, per duplicated name and per `(stage, target)` pair; both tool-loop deliveries answer "what is pending" the same way; a memory bank no longer sends a table name to a backend that has no table; and `AddBankRecordTool`'s documented constructor arguments are the real ones

#### Documented
- the subflow guide says which of a subflow's own config is live inside a push — the rule is by level rather than by field, with wizard-level `settings:` the exception, and `navigation` called out because it is the one word appearing at both levels while the levels disagree. The push/pop lifecycle table now names the end stage's render and its order
- `WizardFSM.stages` documented a stronger guarantee than it delivers: the copy is shallow, so the natural `for name, meta in fsm.stages.items(): meta[...] = ...` edits the running wizard's configuration for the life of the process. Behaviour is unchanged deliberately — a deep copy measures roughly 2500x the shallow one for a guarantee no caller in this package asks for — and two tests pin the boundary
- the multi-tenancy guides say that the API they document is deprecated. `BotManager` and the `dataknobs_bots.api` singleton helpers warn at runtime, so a sample pasted from either guide raised a `DeprecationWarning` on its first call and neither page mentioned it
- the built-in tool table is the catalog again. It stopped at twelve rows while the catalog held twenty-one, and the prose above it stated twelve as well, so the nine wizard tools were named in neither document. A subset is the worst shape this error can take: an obviously partial list invites the reader to go and look, while a stated count reads as a closed set and the reader concludes the rest do not exist. A workspace guard now compares the table and every documented count against the registry in both directions, so the next tool registered cannot land unlisted
- what a custom `storage_class` actually has to provide — three pages listed implementing `ConversationStorage` and supplying `create(config)` as equal requirements when only the second is checked
- grounded retrieval isolates sources from one another, a guard that was already there, had no test, and was unreachable through the two `dataknobs-data` sources that absorbed their own failures

### dataknobs-fsm [0.4.2]

#### Fixed
- `InMemoryStorage` no longer injects connection parameters the memory backend has no fields for. It set `max_size=1000` and `enable_indexing=True` on every in-memory history store; `AsyncMemoryDatabase` accepts neither, so both were discarded by the config projection and the store was never bounded or indexed by them — while the class documented itself as applying those defaults, which is what made the gap invisible

### dataknobs-legacy [0.2.1]

#### Fixed
- `from dataknobs.<pkg>.<module> import Name` resolves — the import form pre-split code actually contains. Each shim re-exported a modular package's submodules by importing them, which binds them as attributes and is enough for `from dataknobs.structures import tree` but not for the dotted form, because Python resolves a dotted module path through `sys.modules` rather than through the parent's attributes

## Release - 2026-08-19

Three changes run across the workspace in this release.

**Path containment is one shared guard rather than a habit.** `dataknobs-common`
gains `safe_join`, `safe_join_or_raise`, `safe_segment` and `PathAnchor`, and
every site that composed an untrusted name onto a base directory now resolves
through them: config names and `@`-references in config; `$include` / `$import`
chains and `FileSystemResource` in fsm; `DocumentFileRef` and ingest globs in
xization; knowledge-backend keys, wizard subflow names and draft ids in bots.
Two spellings were open nearly everywhere the composition was checked at all —
a `..` segment that walks out, and an absolute value that discards the base
outright, which is the wider of the two rather than a narrower case. Refusals
are now one type, `PathEscapeError`, a `ValueError` subclass, so one `except`
covers what four exception types and four wordings used to. A name addressing a
*subdirectory* is unaffected everywhere — that is how a layout convention is
spelled, and bounding a name is not flattening it. Where a deployment spans
sibling trees deliberately, each guard takes an opt-out that logs the escape at
WARNING rather than switching the check off silently.

**A config naming no backend is no longer indistinguishable from one naming the
default.** Sites across data and bots read the key as
`.get("backend", "memory")` or forwarded a typed default unconditionally, so
the absence was consumed one frame above the only code positioned to report it.
The choice is now forwarded only when a config makes it, and the fallback is
reported — at WARNING where an in-process store that loses everything on
restart is the consequence, at DEBUG where the default is the recommended
answer. `UserStateStoreConfig.backend` and `VectorMemoryConfig.backend` default
to `None`, meaning "not chosen here"; code reading either and expecting a string
must handle it.

**The quality gate now covers the test trees.** Every package's tests are linted
at a ceiling of zero alongside its sources, the mypy ceilings that had drifted
above what the tree measures were re-baselined, and each declined ruff rule
carries a category and a written argument that a test checks. Several rules were
un-declined and their sites fixed. No runtime behaviour changed: the two
published DSL names this reached (`ComplexQuery.AND` / `.OR`) are waived per
line rather than renamed.

### dataknobs-common [3.0.0]

#### Changed
- **Breaking:** `BoundTenantContext`, `PrefixedTenantContext` and `SharedCorpusTenantContext` — and `create_tenant_context` / `tenant_context_from_env` — reject a `tenant_id` that is empty, is `.` or `..`, or contains `/`, `\` or NUL. A separator inside the one segment whose job is isolation merged two tenants' state: by traversal on a filesystem backend, and with no traversal at all on a key-string backend, where the namespaces simply collide. Use a flat identifier; `PrefixedTenantContext`'s `prefix_pattern` stays free-form for a nested convention, and now documents that a pattern picks its own delimiter and this check cannot know what it is
- **Breaking:** `find_config_file` rejects a name containing `..` or an absolute one — both previously addressed outside `config_dir`, and the name reaching it is frequently an `extends:` value, an environment variable, or a resolver's output rather than a caller's literal. `allow_outside=True` opts back out, logging each real escape at WARNING
- `PluginRegistry.get_metadata()` returns a deep copy; a caller reading and editing a nested value previously changed what every later caller saw
- `PluginRegistry.create()` reports a routing key it supplied itself — at WARNING for the three registries whose default fails silently (a lock every process holds at once, a rate limit enforced once per process, a bus whose events reach nobody), at DEBUG where the default is the recommended answer
- `validate_tenant_id` is `safe_segment` under a different name — same rule, same rejections, and it raises `SegmentEscapeError`, which is still a `ValueError`, so no caller changes
- `ConfigPathEscapeError` also subclasses `PathEscapeError`, so one `except PathEscapeError` reaches every composing site. Purely additive; it keeps `ConfigLoadError`
- the Ollama probes and markers take `host` / `port`. Existing calls are unaffected, but `None` now means "resolve from the environment" where `is_ollama_model_usable` meant `localhost`

#### Added
- `FileLock` (`dataknobs_common.locks`) — a path-keyed advisory lock excluding every overlapping holder, which needs both halves: `fcntl.lockf` / `msvcrt.locking` across processes and a mutex per lockfile inode within one, since POSIX record locks are owned by the process. Keyed off the file rather than the string naming it, so a symlink and its target share one lock. Not a `DistributedLock`, and not reentrant
- `safe_join` / `safe_join_or_raise` / `PathEscapeError` — compose a path from untrusted parts without leaving a base, judged lexically so it is safe on an event loop and works on a path that does not exist yet
- `safe_segment` / `SegmentEscapeError` — the second question a composed name has to answer. `safe_join` asks *may this address here?*; this asks *is this one segment?*, which is what an identifier interpolated into a layout with literal segments of its own has to be
- `PathAnchor`, with `anchored_at()` / `rooted_at()` constructors — a boundary fixed when a load starts while the position inside it moves, for a loader following references from file to file
- `validate_tenant_id` is exported from `dataknobs_common.tenancy`, for a consumer writing its own `TenantContext` impl — nothing runs `__post_init__` on a class this module never sees
- `PluginRegistry.declare_unavailable()`, `is_known()`, `list_known_keys()`, `list_canonical_keys()`, `load_declared_type()`, `get_metadata(follow_alias=True)` and `PluginRegistry(default_warning=...)`. A plugin behind an optional dependency has three states rather than two, and a registry holding only factories reports "not installed" as "misspelled" — losing the one answer worth having, which is what to install
- `requires_real_postgres` / `requires_real_postgres_sync` / `requires_real_elasticsearch` / `requires_real_s3`, and the `must_skip_real_service()` predicate behind them — a gate testing all three terms a behavioural suite depends on (opted in, reachable, driver installed) rather than only whether a server answered
- `ollama_env_params()`, `list_ollama_models()` and `wait_for_ollama()` in `dataknobs_common.testing`

#### Fixed
- the Ollama probes shelled out to the local `ollama` CLI, so a machine without the binary reported a reachable server down — a silent skip of every gated test rather than a failure. They probe `GET /api/tags` now, match model names instead of rendered table text (`mistral` was satisfied by `mistral-small`, and `GB` reported available), understand all three spellings of `$OLLAMA_HOST`, and bound the response body so a misdirected variable cannot hang the probe
- a populator that replaced the default factory and then failed kept the replacement, so the next `create()` for an unregistered key silently succeeded off it
- `unregister()` stranded a withdrawn plugin's aliases, leaving them answering `{}` to the one question a withdrawn plugin stays visible to answer
- a literal `%` in `default_warning` raised inside `logging`, at the first fallback, naming neither the registry nor the text

#### Security
- `safe_join` accepted a NUL inside a path part. Nothing was exploitable through it, but the refusal arrived from a different place as a different type than every other rejection of a bad name

### dataknobs-config [0.6.0]

#### Changed
- **Breaking:** `$requires` on a resource the environment does not define now raises. The severity was inverted — a resource that existed but lacked a declared capability aborted the build, while one that did not exist at all resolved to its inline defaults and reached the factory. Declare `$required: false` alongside `$requires` where "if it is there it must do X; it may be absent" was the intent
- **Breaking:** a `$`-prefixed key that is not a marker is an error. The set is closed, and the comprehension building a reference's inline defaults took everything else — so `$requred: true` was promoted to a default and passed to the factory as a keyword argument, meaning *not required*, at the exact site meant to close that class of failure. `$requires` must be a list (a bare string iterates character by character), and `$required` / `$requires` on a block with no `$resource` is rejected, which is what gives away a typo in the selector key itself
- **Breaking:** an `@`-reference is contained within `config_root`. Any config *value* beginning with `@` is a file reference, so a `..` segment climbed out of the tree and the absolute branch never consulted `config_root` at all. `Config(..., allow_reference_outside_config_root=True)` is the migration — a caller argument, deliberately not settable from a config file, since a reference is bounded precisely because it comes out of config content
- **Breaking:** `EnvironmentConfig.load()` raises on an escaping environment name where it previously returned an empty config, so a deployment with a malformed `DATAKNOBS_ENVIRONMENT` booted on defaults. A name merely *absent* from `config_dir` still returns an empty config
- **Breaking:** `resolve_for_build(strict_resources=..., resolve_resources=False)` raises `ValueError`. The pair validated nothing and returned a config anyway, from the method documented as the startup preflight
- `ConfigBindingResolver` resolves references nested inside the resource it looks up, rather than passing one on as a literal `{"$resource": ...}` keyword argument. Both resolvers now run the same resolution below the entry point
- `ResourceNotFoundError` subclasses both `EnvironmentConfigError` and `KeyError`. `resolve_for_build()` could not previously raise a `KeyError`; under a strict policy it can, so an `except KeyError` there for unrelated reasons will swallow it. Its `__str__` is restored — `KeyError`'s wrapped the message in quotes and escaped every name inside it

#### Added
- **A `$resource` reference can declare that its resource must exist**, at four levels, most specific first and each unset-means-defer: `$required: true` on the reference, a non-empty `$requires`, `strict_resources=` on `resolve_for_build` / `EnvironmentAwareConfig`, and `settings: {strict_resources: true}` on the environment. The default is unchanged — a missing resource still warns and degrades. The environment level is the only one a deployment whose references are generated at runtime can reach
- `find_unresolved_resources()` on `EnvironmentAwareConfig` — every unresolvable reference in one pass as `UnresolvedResourceRef(path, resource_type, resource_name, required, has_inline_defaults)`. Raise-on-first is right for a build and wrong for a preflight. It runs the *same* walk as the build, so it is a prediction of it rather than a second opinion
- `resolve_resource_references(config, environment, ...)` is exported, with `RESOURCE_MARKER_KEYS`, `STRICT_RESOURCES_SETTING` and `UnresolvedResourceRef` — the shared primitive, so a consumer holding a config tree does not become another reader of the format. Reading it independently is what produced the divergences this release closes
- `EnvironmentConfig.get_resource(..., required=)` separates data from policy. `defaults` carried both meanings at once, which made one combination unreachable: merge these values, but still fail if the resource is absent
- `allow_outside=True` on `InheritableConfigLoader`, `EnvironmentConfig.load()` and `EnvironmentAwareConfig.load_app()`, for a layout that genuinely spans sibling trees. Off by default; a real escape logs at WARNING
- a reference cycle raises `ConfigError` naming the chain, in both the build and the survey, and failure messages name the dotted config path of the reference that failed

#### Security
- a configuration name could address a file outside the directory it was loaded from, on all three loaders. Three of the four names this affects are not a caller's own literal: an `extends:` value read out of a config file, an environment name from `DATAKNOBS_ENVIRONMENT`, and a consumer-supplied resolver's output

### dataknobs-utils [2.0.1]

#### Changed
- **Integer and float columns are typed by range and width.** `integer` is a 4-byte signed type while pandas defaults to `int64`, so a column holding a value past 2³¹ created a column its own data could not enter; `real` carries ~7 significant digits, so a `float64` round-tripped silently rounded. 64-bit maps to `bigint` / `double precision`, `uint64` to `numeric(20)`, and genuinely narrow columns keep the narrow type. Note the interaction with `CREATE TABLE IF NOT EXISTS`: a table already created with the narrower column keeps it
- the `varchar` width is measured on the rendered value with the renderer that sends it, and a column of empty strings emits `varchar(1)` rather than the `varchar(0)` PostgreSQL refuses outright

#### Fixed
- **`PostgresDB` emitted a `CREATE TABLE` that crashed on boolean and timestamp columns.** `_psql_schema_line` named only integer and float and fell through to `.str.len()` for everything else, so a `bool`, `datetime64[ns]`, nullable `boolean` or tz-aware column raised `AttributeError` and the table could not be created at all. The ladder now maps `bool` → `boolean`, `datetime64` → `timestamp`, tz-aware → `timestamptz`, `timedelta64` → `interval`
- **`PostgresRecordFetcher.get_records` inlined three identifiers unquoted, one of them caller-supplied per call.** `fields_to_retrieve` was a reachable injection vector rather than a hardening gap — a fetcher configured for one table returned a column from another, plus `current_user`, through nothing but that parameter. All three positions are quoted, `ids` is bound, and `table_head`'s `LIMIT` is bound
- **`PostgresDB` never closed a connection and opened a new one per call**, measured at 79% of wall time for a trivial `SELECT 1`. `DotenvPostgresConnector` holds one connection per thread — per thread because psycopg2's `with conn` is not re-entrant and two threads would share a transaction. `PostgresDB` gains `close()` and context-manager support; a cached connection is validated with a round trip, since `connection.closed` reports only what this process did to it
- `upload` sent every value as text, so a null arrived as `'nan'`, a nullable `Int64` upcast to `'1.0'`, and a `timedelta64[ns]` as `'86400000000000 nanoseconds'`. Values are gathered per column and passed as typed parameters
- `upload` built a syntactically invalid INSERT for a frame with no rows, and rejected default integer column labels with a message naming neither the subject nor the fix

### dataknobs-xization [2.1.0]

#### Security
- **Breaking:** a glob pattern could enumerate outside a `LocalDocumentSource`'s root. `Path.glob` treats `..` as an ordinary literal segment, so `../secrets/*.env` yielded refs carrying each file's real size and resolved absolute `source_uri` onward into chunk metadata. The claim that no check was needed — that every ref is derived with `relative_to(root)` — does not hold: `relative_to` re-expresses a path lexically and enforces nothing. A match outside the root is now skipped and logged. Breaking for a configuration whose patterns deliberately reached outside the root; such an ingest previously enumerated those files and then failed on the first read
- a `DocumentFileRef` can no longer read outside that root either. `read_bytes` and `read_streaming` composed `root / ref.path` unchecked, and an absolute `ref.path` discarded the root. The `DocumentSource` protocol states the rule, so a consumer-written implementation inherits it

### dataknobs-data [0.9.0]

#### Changed
- **Breaking:** a configured `domain_id` scopes the id-keyed operations too. `get_vectors()`, `delete_vectors()`, `update_metadata()`, `add_vectors()`, `add_documents()` and `metadata_fields()` address rows by id and so built no filter, which left the tenant scope binding only the surfaces that take one — a scoped store answered for any id in the collection, and `metadata_fields()` returned the union of every tenant's key names. An out-of-domain id is now answered exactly as an absent one. Code that used a scoped store to reach outside its domain gets `(None, None)` and no effect; unscoped stores are unaffected
- **Breaking:** `ChromaVectorStore.update_metadata()` replaces a row's metadata instead of merging into it, matching the other three backends and the contract the ABC now states outright. Code relying on the merge for a partial update must supply the full dict, or use `update_metadata_where()`, whose contract is a merge and is unchanged. The same methods return rows *matched* rather than rows written
- **Breaking:** a file `persist_path` written by two overlapping instances raises `ConcurrencyError` rather than silently discarding one of them — `save()` serializes the whole in-memory state, so the earlier writer's rows were gone from disk entirely. Covers `FaissVectorStore` and `MemoryVectorStore` alike, on `VectorStoreBase`. Three consequences: `close()` persists only a store that was mutated (load-bearing, since a no-op write would move the file's identity); `save(force=True)` overwrites deliberately and is the way out of a refusal; and the check is best-effort — mtime, size and inode — so it catches the common accident and is not a lock
- **Breaking:** `FileLock` is no longer reentrant — one thread acquiring the same path twice deadlocks. It worked only because `fcntl` grants the owning process a lock it already holds, which is the defect the new intra-process mutex fixes; there was no way to keep it without keeping the hole. `from dataknobs_data.backends.file import FileLock` still resolves, to `dataknobs_common.locks.FileLock`
- **Breaking:** `VectorStoreFactory.create()` refuses a backend whose driver is absent *before* construction, with the same `ValueError` the database factories raise, rather than building the store and regexing the resulting `ImportError` for a `pip install` line. Code matching the old text needs updating; code catching `ValueError` does not
- `domain_id=""` scopes on every backend. `PgVectorStore` guarded on truthiness while the metadata-carrying backends tested `is None`, so an empty-string domain isolated on three backends and ran completely unscoped on the fourth
- an empty `add_vectors()` batch is a no-op everywhere. The four disagreed and one corrupted the store — `MemoryVectorStore` minted an id for a zero-dimension vector and grew by a row
- `SyncFileDatabase` and `AsyncFileDatabase` leave a `<path>.lock` beside their data file. Removing it on release is what let two holders in, and two instances in one process are genuinely serialized now
- a persisted file keeps the permissions it had; the publish used to reset the mode to the umask default on every save
- a post-filtered Chroma search escalates its fetch rather than settling for one over-fetch, doubling up to the whole collection where `count()` can bound it, so the answer becomes exact. Declaring the key in `scalar_metadata_keys` pushes the predicate down and avoids the round-trips
- backend-selection log records come from `dataknobs_data.backend_selection`, not from `dataknobs_data.factory` / `vector.stores.factory` — a consumer routing by logger name needs the new one
- a `vector_store` section naming an uninstalled backend still validates. Whether a config is well-formed does not depend on which optional drivers the machine reading it happens to have
- `backend: null`, `backend: ""` and a non-string say which way they are unusable, instead of rendering into `Unknown backend type: <value>` and sending the reader after a spelling mistake

#### Added
- `dataknobs_data.backend_selection` — `select_backend()`, `available_backends()`, `backend_available()`, `backend_info()`, `normalize_backend()`, `register_backend()`, `build_backend()` and `DEFAULT_BACKEND`, one resolution shared by the three factories and by `AsyncDatabase.from_backend()` / `SyncDatabase.from_backend()`, which held a fourth copy
- `compose_scope_key()` — the three-case scope composition (absent key, in-scope value, out-of-scope value) as a module-level function beside the contract it depends on. The obvious spelling, overwriting the caller's key, is wrong in the one direction that costs data: it turns a request for *another* scope into a request for *this* one, so a destructive call that should match no rows matches every row in the caller's own scope
- `ChromaVectorStore` tracks `created_at` / `updated_at`, exposed via `include_timestamps=True` on `get_vectors()` and `search()` — the surface the other three backends already carried. A `timestamps:` block now takes effect where it was previously parsed and never read
- `get_available_backends()` / `is_backend_available()` on all three factories, `get_backend_info()` on `AsyncDatabaseFactory`, and backends that stay described when their driver is missing, so `requires_install` answers in the one state anyone asks it
- `dataknobs_data.testing` — deterministic vector draws, for this package's tests and for consumers testing their own `VectorStore`

#### Fixed
- **`FaissVectorStore.search(filter=...)` applied the filter after the index had truncated to `k`**, so a filtered search returned only the matching rows that happened to fall inside the global top-`k` window. `ChromaVectorStore.search()` under-returned for the same reason at a wider window
- **`update_vectors()` reset a row's `created_at` and destroyed rows on a refused batch.** It was `delete_vectors()` followed by `add_vectors()`, and the delete bought nothing
- **a scoped write could capture another domain's row by writing its id**, and `update_metadata()` could push a row out of its own domain, since on three backends the configured `domain_id` lives *in* the metadata dict that method replaces wholesale
- a row whose `domain_id` is a list belongs to every domain named, and is no longer invisible to half its own store or narrowed by a scoped write
- `ChromaVectorStore` alone: `add_vectors()` silently discarded a write to an id it already held; `add_documents()` did not apply the configured `domain_id`; `search_documents()` scored every store as though it were cosine; `search()` and `get_vectors()` raised `TypeError` on `include_timestamps`; and a consumer value under a reserved timestamp key could become a row's creation date
- consumer metadata is no longer shared between a store and its caller in either direction — on Memory and FAISS a caller could edit a stored row without calling a mutator
- the single-file persistence path is fixed throughout: the scratch sweep read its target's name as a glob pattern, `fsync` before publishing was a no-op on Windows, a symlinked `persist_path` was replaced rather than written through, concurrent publishes collided on one scratch file, a `load()` could run inside the store's own `save()`, a half-landed FAISS publish left the store refusing every save, and a read-only directory refused to load
- `SyncPostgresDatabase.close()` closed nothing — the comment giving the reason was false — and `connect()` replaced its `PostgresDB` without closing the first
- `VectorStore.get_vectors()` is annotated to return what it returns: every backend yields `(None, None)` for an id it does not hold

### dataknobs-llm [0.7.1]

#### Fixed
- two docstring examples that could not run, both reaching for a vector capability through `database_factory` and inventing a backend name for it. The `AsyncLLMProvider.embed` block is gone rather than corrected — it taught another package's storage API from inside the embedding API's docstring, which is how it came to be wrong in three ways without anything noticing; `VectorStoreFactory` is named in `See Also` instead

### dataknobs-bots [0.11.0]

#### Changed
- **Breaking:** `InMemoryBotRegistry` and `create_memory_registry` take keyword arguments only, forwarding to `BotRegistry` instead of re-declaring its seven parameters. Deliberate rather than incidental: `BotRegistry` takes `backend` first while the subclass did not, so inheriting the signature would have made a positional first argument mean something else entirely, silently. Every documented example already uses keywords
- **Breaking:** an S3 content key is normalised, so a non-canonically-spelled object written by an earlier release is unreachable — `sub/../guide.md` now composes `{prefix}acme/content/guide.md`. Two spellings of one intended file were two distinct objects, which is the defect, but an existing bucket holding such an object needs it re-keyed. Affects only buckets written with non-canonical or escaping paths
- **Breaking:** an empty `domain_id` is refused on every knowledge-backend method, not only `key_pattern`. `None` remains the all-domains spelling; an empty string now means a caller passed an unset variable, which is what it always was
- `ConfigCachingManager` resolves `$resource` references through `dataknobs-config` instead of walking the config itself. It was a third reader of the format, recognising `$resource` and `type` and nothing else — so it discarded every inline default, ignored `$required` and `$requires`, passed a misspelled marker on as data, and left a nested reference as a literal dict. Raising on a resource the environment does not define is the behaviour it has always had and is preserved
- `ConfigValidator` reports a misspelled marker in a `$resource` section. Skipping every `$`-prefixed key let `$requred: true` — which reads as *not required* — past the one check that runs before resolution
- a non-canonical `PrefixedTenantContext` pattern fails every S3 state call rather than only the file backend's

#### Added
- **every entry point from a portable config to a running bot takes `strict_resources`** — `DynaBot.from_environment_aware_config`, `BotManager`, `BotRegistry`, `ConfigCachingManager`, and `InMemoryBotRegistry` / `create_memory_registry` through the keyword forwarding above — so an operator can state the policy wherever the bot is built rather than at one of them. `ConfigCachingManager` takes it defaulting to `True`, which is what it has always done; the difference is that the posture was a literal, so an operator who had explicitly written `strict_resources: false` was overridden by it

#### Removed
- **Breaking:** `ReviewArtifactTool`, `RunAllReviewsTool` and `GetReviewResultsTool`, which could not be called successfully — `registry.get()` had become async and was still called without `await`, and `add_review()` / `get_definition()` do not exist on `ArtifactRegistry` at all. The un-awaited `get` was the quietest: a coroutine is truthy, so the `if not artifact` guard passed and a coroutine travelled on in place of an `Artifact`. The module's whole suite was skipped under a note saying the tools used the old API, so three broken public classes sat behind a passing build. No working caller can exist. The capability was superseded by `ArtifactRegistry.submit_for_review()` and `get_evaluations()`

#### Fixed
- **two knowledge bases over one vector store overwrote each other's chunks.** `domain_id` folds into the chunk-id prefix, but nothing supplied the value unless a `KnowledgeIngestionManager` threaded it through, so the common shape — a shared store, one knowledge base per bot — collided. A knowledge base now derives its binding from the store it is bound to, and the binding means one thing at every surface
- **a bound `tenant_id` now scopes `count()`, `clear()` and `update_metadata_where()`.** It scoped `query` and `hybrid_query` and nothing else, so on a shared store `clear()` reached every tenant's rows
- **`AutoIngestionMixin` forwards the whole knowledge-base config to the ingest knowledge base.** It hand-copied six keys while the bot's own knowledge base is built from the entire section, so the two disagreed about everything the whitelist did not name — `tenant_id` among them, which meant the ingest wrote untagged chunks that the bot's tenant-scoped reads could never match: a total retrieval blackout reported as a successful ingest. The projection is now a pass-through with a named exclusion set
- replacing a knowledge base's vector store no longer orphans the chunks it already wrote, and `set_provider()` no longer inverts the close-ownership gate on `RAGKnowledgeBase`, `VectorMemory` and `SummaryMemory`
- an empty-string binding is a binding at every surface — the chunk-id fold tested truthiness while identity stamping and filter composition tested `is not None`
- a knowledge-base config overriding the registration's domain now says so at WARNING. The precedence is right for a section written for one bot and is unchanged, but the same section is routinely reused as a template, and then it quietly points every domain at one namespace
- `embedding_base_url` reaches the embedder on the legacy flat config shape; it was forwarded under a name no config field carries and discarded in silence
- a wizard skipped extraction after an auto-advance under `advance()` and never under a conversation
- every path-containment refusal is one catchable type, where `FileKnowledgeBackend`, `ConfigDraftManager` and `WizardConfigLoader` each raised a bare `ValueError`
- `key_pattern()` named the wrong document in a tenanted deployment, and the watch built from it looked healthy

#### Security
- **deleting one knowledge base destroyed every tenant's ingest state.** A tenant context contributes a prefix to every state key, and that prefix's first segment landed at the same level as a knowledge base's own domain segment
- **a knowledge base could overwrite and delete another one, on every persistent backend.** `domain_id` is interleaved with the layout's own literal segments — `{domain}/content/{path}`, `{domain}/_metadata.json` — so an identifier occupying more than one slot addressed a neighbour's storage. One tenant could also read another's ingest state through the two-hop file-backend composition
- a knowledge-base `domain_id` or resource path, a wizard subflow name, a `wizard_config` path, and a `ConfigDraftManager` config name or draft id could each address a file outside their declared base. `S3KnowledgeBackend` composed every key unchecked, its file-backend twin having been guarded a release earlier

### dataknobs-fsm [0.4.1]

#### Security
- **a `$include` or `$import` could read any file on the volume.** `ConfigLoader._resolve_references` composed the reference onto a base directory and opened the result unchecked at both sites, and because the loader rebases to each included file's parent before recursing, a chain walked wherever the first hop reached. Containment is now judged against the config tree, fixed once when the load starts, so a fragment in a subdirectory may still reach a sibling above it. `load_from_file(config_root=...)` widens the anchor for a deployment whose configs deliberately span sibling directories — widening rather than switching the check off keeps the boundary a boundary
- **a path handed to `FileSystemResource` could address any file on the volume.** `__init__` resolves `base_path`, which is only meaningful if that directory is a boundary, but no composed path was ever checked back against it — `acquire`/`open`, `exists`, `delete` and `list_files` each composed independently. Two placements are load-bearing: in `delete` the check runs before the blanket `except Exception: return False`, since `False` is also that method's "no such file" answer, and in `acquire` the refusal is re-raised ahead of the handler that would otherwise mark the resource `ERROR` for a caller's bad name

#### Fixed
- a cyclic `$include` recursed until the interpreter gave out, so a one-character typo in a fragment surfaced as an apparent interpreter fault with a thousand-frame traceback naming no file. A cycle raises `ConfigLoadError` naming the chain, tracking the files currently *open* rather than those already seen, so an ordinary shared fragment is unaffected
- a malformed reference raised a bare `KeyError('file')` or a `pathlib` error, naming neither the directive nor the file that carried it
- the include cache outlived the load that filled it, so a `ConfigLoader` held across loads — the ordinary way to hold one — served the first load's copy of every fragment however the file had changed

#### Added
- behavioural tests for `$include` and `$import`, which had none in this package's suite or in any example config in the repository, so the containment work above cannot buy a boundary by breaking the feature

## Release - 2026-08-11

Two changes run across the workspace in this release.

**Error messages no longer relay the text of an underlying failure.** Where a
message previously interpolated an exception raised by a driver, a parser, an
imported module, or a provider's SDK, it now names what failed and the
exception type, with the original on `__cause__` and in the logs. That text was
unbounded — a connection URL, an absolute filesystem path, a quoted config line
containing a credential — and some of these types are rendered with their
message shown at an HTTP boundary. Types, attributes, and `retry_after` are
unchanged; anyone matching on error *text* rather than on its *type* is
affected, in bots, common, config, fsm, llm, and xization.

**Dotted paths resolve through one shared resolver**
(`dataknobs_common.imports`), replacing nine copies that disagreed. Every
config key taking a dotted path now accepts both `module:Name` and
`module.Name`, where some keys previously took only one; only paths that were
already rejected start resolving, so no working configuration changes meaning.
Failures raise `DottedPathError` or `DottedPathTypeError` — both
`ConfigurationError` subclasses — rather than an assortment of `ValueError`,
`ImportError`, `AttributeError`, `KeyError`, and `TypeError`. A caller catching
`ConfigurationError` is unaffected; one catching a specific stdlib type is not.

### dataknobs-common [2.0.0]

#### Changed
- **Breaking:** `assert_no_broad_except_in_error_text` treats `ImportError` as unbounded by default — its own text carries an absolute site-packages path. `unbounded_types=` still **replaces** the default set rather than extending it

#### Added
- `dataknobs_common.packs` — ordered, precedence-resolved composition of named declaration bundles, with per-field merge rules a domain package declares for itself
- `aclose_if_owned` — the third owned-vs-injected close guard, for a collaborator carrying both a sync `close()` and an `aclose()`
- `assert_no_leaked_bridge_threads` — fails when a block leaks a dataknobs daemon thread, measured as a delta so an earlier test's leak cannot name the wrong culprit

#### Fixed
- a `SyncLoopBridge.close()` that raised left every later caller waiting forever
- `PostgresEventBus.from_config` / `PostgresAdvisoryLock.from_config` config handling

### dataknobs-config [0.5.0]

#### Changed
- **Breaking:** `EnvironmentConfig.merge()` can raise `RequiredEnvVarError`. It was previously pure data manipulation that never read the environment; it now normalizes mixed substitution provenance. Merging two sides that agree still cannot raise
- **Breaking:** `InheritableConfigLoader.load(use_cache=False)` no longer writes to the cache, and `clear_cache(name)` now transitively clears dependents
- `get_resource()`, `merge()`, and `to_dict()` copy nested containers instead of handing back the environment's own objects
- a `$resource` name or `type` containing `${VAR}` now resolves, instead of matching nothing and falling back to inline defaults

### dataknobs-structures [1.0.16]

#### Added
- a package changelog, with per-version history reconstructed from the release tags back to 1.0.0

#### Changed
- sources adopted strict typing at a ceiling of zero findings; the one outstanding finding was fixed rather than waived

### dataknobs-utils [2.0.0]

#### Fixed
- **Breaking:** `RequestHelper.get` / `post` / `put` / `delete` / `head` sent requests with **no timeout at all** — the wrappers spell "unset" as `None`, which reaches `requests` as *wait indefinitely*, on calls the caller believed carried the configured default. All five now fall back correctly
- `load_project_vars` dropped `None` values from bare `KEY` lines rather than raising `TypeError` when setting the environment
- raised the `nltk` floor to `>=3.10.2`, excluding the broken 3.10.1 import hook

### dataknobs-xization [2.0.0]

#### Changed
- **Breaking:** chunker and transform path resolution raises `DottedPathError` / `DottedPathTypeError` instead of `ImportError`, `AttributeError`, or `TypeError` — catch `ConfigurationError`
- a `chunker:` or transform key written `module.path:Name` resolves, instead of falling through to a registry lookup and reporting an unregistered plugin

### dataknobs-data [0.8.0]

#### Added
- `[chroma]`, `[faiss]`, and `[pgvector]` extras splitting the all-or-nothing `vector` extra, so reaching FAISS or pgvector no longer pulls chromadb and its unfixed advisory. `[vector]` is retained as a roll-up and resolves to the same distributions as before

#### Changed
- `ConversionOptions.merge_metadata` documents list-replace and delegates to the shared `deep_merge`; behavior unchanged

### dataknobs-legacy [0.2.0]

#### Removed
- **Breaking:** `dataknobs.flask_api`, and with it the `flask` dependency it alone required. The module imported a `create_app` this package does not define, so `import dataknobs.flask_api` already raised `ImportError` <!-- dk-imports: illustrative -- a Removed entry names what is gone; the name resolving would falsify it -->

### dataknobs-llm [0.7.0]

#### Changed
- **Breaking:** `_dataknobs_error_for_status`'s second parameter is renamed `message` → `detail`. An out-of-tree provider passing `message=` gets a `TypeError`; one passing it positionally keeps working but has its string read as classification material and discarded. A provider can no longer influence a translated error's message
- `ConversationManager` persists the canonical provider family key on assistant-node metadata, where it previously stored `config.provider` verbatim
- the missing-aiohttp `ImportError` points at the floor-governed extras rather than an unconstrained `pip install aiohttp`

#### Added
- `CostCalculator.cost_from_tokens(pricing, input_tokens, output_tokens)` — pricing for callers holding token counts rather than an `LLMResponse`
- `LLMProviderFactory.list_providers()` — every registered family key, reflecting consumer registrations

#### Security
- raised the `aiohttp` floor to `>=3.14.3` (extras: `ollama`, `huggingface`)

### dataknobs-bots [0.10.0]

#### Changed
- **Breaking:** a config key naming something that cannot be imported is now fatal, where four classes of key were a warning and a bot that started while quietly doing less than its configuration said — derivation rules, wizard hooks, turn-lifecycle hooks, task-injection hooks, and a wrong-*type* `context_transform`. All faults in a block are reported together, and nothing is registered from a block containing one. Every case logged a WARNING first, so existing logs show whether a deployment is affected
- **Breaking:** the `provider` value in turn logs, in `after_message` middleware kwargs, and across every cost-stats surface (`by_provider` in `get_client_stats()` / `get_all_stats()`, and the `export_stats_json()` / `export_stats_csv()` buckets) is now the canonical family key — `"openai"` where it read `"OpenAIProvider"`. The class name moved to `TurnState.provider_impl` and the `provider_impl` log field
- **Breaking:** recorded costs change. Pricing resolves from the provider's own model profile before the built-in table, and a dated-model-ID lookup bug billed `gpt-4o-mini-*` at `gpt-4o` rates
- `api.RateLimitError` and `api.BotCreationError` are now also `OperationError`, widening what an existing `except OperationError` block catches
- handled DataKnobs errors no longer propagate to the ASGI server, so a deployment alerting on unhandled exceptions sees that signal drop; the handlers log every error they handle instead

#### Security
- **Breaking:** `BotCreationError` no longer returns its `reason` to the caller. A subclass setting `client_safe = True` restores the old behavior; the other API error classes are unaffected
- per-instance `cost_rates=` overrides permanently rewrote the class-level defaults, so one middleware instance's rates repriced every instance built afterwards in the same process — including other tenants'
- declared an `http` extra pinning `aiohttp>=3.14.3` for `HTTPRegistryBackend`, whose transport reached consumers only transitively and under no dataknobs floor

### dataknobs-fsm [0.4.0]

#### Changed
- **Breaking:** `CircuitBreakerError` is a `ResourceError`, not a `ConcurrencyError`. `except ConcurrencyError` no longer catches it, and it maps to 503 rather than 409 — retry logic keyed on the old base treated an open breaker as a contended write worth re-attempting at once
- the `functions.base` exceptions join the shared hierarchy, so `except DataknobsError` reaches 60 raise sites it previously missed and a boundary resolves them to 503 / 422 rather than an indistinguishable 500. `except FSMError` catches exactly what it caught before
- `CircuitBreakerError.retry_after` answers alongside the existing `wait_time`
- the `llm` and `vector_store` resource types report as unsupported instead of failing on an internal import naming a module that does not exist

#### Deprecated
- `functions.base.FSMError`, `ConfigurationError`, and `StateTransitionError` (with its `FunctionError` alias) — the three that duplicate a `core.exceptions` name and that nothing in the package raises

#### Fixed
- **Breaking:** `ConfigLoader.merge_configs` replaces list-valued fields instead of extending them, matching its own docstring. An FSM's substance is list-shaped, so two configurations each declaring a network named `main` previously merged into two networks both named `main`
- `merge_configs` no longer overrides fields the later configuration never mentioned
- `AdvancedFSM.aclose()` stalled the caller's event loop; `SimpleFSM.aclose()` and the CLI leaked daemon threads
- `ResourcePool.acquire()` waited out its whole timeout before creating a resource, and ignored `timeout=0`

#### Security
- raised the `pymdown-extensions` floor to `>=11.0.1`

## Release - 2026-07-29 (2)

### dataknobs-bots [0.9.4]

#### Fixed
- platform middleware injection

### dataknobs-llm [0.6.9]

#### Changed
- maintenance release

## Release - 2026-07-29

### dataknobs-bots [0.9.2]

#### Fixed
- scope wizard undo/rewind bank revert per conversation
- Scope WizardReasoning memory banks per conversation
- harden wizard close cascade + restore bank lifecycle
- add database teardown to ArtifactBank and ArtifactBankCatalog
- type AsyncMemoryBank db params as AsyncDatabase
- AsyncMemoryBank database lifecycle parity + from_dict leak

### dataknobs-common [1.6.3]

#### Added
- consent-gated access for per-user state sections

### dataknobs-config [0.4.4]

#### Changed
- maintenance release

### dataknobs-structures [1.0.15]

#### Changed
- maintenance release

### dataknobs-utils [1.2.18]

#### Changed
- maintenance release

### dataknobs-xization [1.3.14]

#### Changed
- maintenance release

### dataknobs-data [0.7.0]

#### Added
- add section schema versioning with lazy on-read migration for per-user state
- add a persisted append-only audit log for per-user state
- per-section prune attribution + deletion-event hardening
- emit a delta event on per-user state deletion and erasure
- retention pruning for per-user state sections
- consent-gated access for per-user state sections
- route upsert mint fallback through _generate_id() hook
- route create() mint through one overridable _generate_id() hook

#### Changed
- maintenance release

#### Fixed
- harden on-read migration payload isolation + version validation
- make per-user event-log append truly best-effort
- harden retention pruning (deep-review findings)
- lock the reserved consent section out of the content API
- upsert no longer mutates the caller's record
- mint id-less bulk writes; record-id docs + S3 no-block test

### dataknobs-fsm [0.3.3]

#### Changed
- maintenance release


## Release - 2026-07-27

### dataknobs-common [1.6.2]

#### Added
- BoundedLRUCache primitive for bounding a per-key in-memory cache (pin/unpin, on_evict hook)
- is_ollama_model_usable canary + requires_ollama_usable_model pytest marker

#### Fixed
- reclaim pins on BoundedLRUCache manual removal

### dataknobs-llm [0.6.8]

#### Added
- unified model-metadata substrate (model_profile) with per-provider ModelConstraints
- LiveApiSource — generic live-vendor-API model-metadata source, with an injectable match= seam
- ConfigOverrideSource gains an injectable match= matcher
- OpenAI, Bedrock, Ollama, and HuggingFace model-metadata bindings
- provider pricing/cost accessors + model_limits tooling --provider flag
- LLMConfig.model_profile_overrides loose-mapping field
- Claude 5 family (Opus 5 and siblings) model support
- ConversationManager.reset() — roll a conversation back to its empty pre-message state
- token-budget truncation signal (LLMResponse.truncated) surfaced cross-provider
- distinct ContextLengthExceededError on context-window overflow
- in-loop conversation history compaction + shared summarization seam
- configurable mid-conversation system-message policy for Anthropic
- shared message-sequence utility + normalize_claude_stop_reason() helper
- EchoProvider.set_response_delay for scripted latency

#### Changed
- finish_reason is now the canonical cross-provider vocabulary (stop / length / …)
- vendor API errors now raise dataknobs_common.exceptions types across all providers, incl. mid-stream
- Anthropic live Models-API ceiling cache is now per-provider-instance
- Anthropic mid-conversation role="system" messages default to inline

#### Fixed
- clamp max_tokens to the model's output ceiling; reject over-ceiling requests
- shape OpenAI/Bedrock requests to the model family's rules (param drop/clamp/remap)
- validate_model no longer rejects current models; replace stale whitelists
- honor model_profile_overrides pricing/available overrides
- Claude 5 temperature-rejection list completeness
- TokenCounter.estimate_tokens / estimate_messages_tokens tolerate None
- keep tool_result blocks first when inlining mid-conversation system messages

### dataknobs-bots [0.9.2]

#### Added
- bounded conversation-manager cache (max_cached_conversations)
- bounded per-conversation undo history (max_undo_checkpoints)
- structured ReAct termination reason
- opt-in in-loop history compaction for the ReAct strategy
- DynaBotConfig.tool_loop_timeout_message user-facing text
- ReActReasoningConfig.truncation_retry_max_tokens opt-in adaptive-budget retry

#### Changed
- unify buffered and streaming tool-execution loops onto one core

#### Fixed
- wizard collection-mode records are now revertable by undo
- fully revert strategy/bank/tree state when undoing back through the first turn
- distinguish an emptied conversation on undo/rewind
- rewind_to_turn to the current turn is now a no-op
- clear_conversation reclaims a conversation's undo checkpoints
- bound the terminal synthesis of a phased reasoning turn by the remaining budget
- treat a truncated tool call as terminal, not executed; pair orphan tool_use at finalize

### dataknobs-fsm [0.3.2]

#### Security
- bump pymdown-extensions floor to >=11.0.0 (docs dev dependency; CVE-2026-61632)

## Release - 2026-07-20

### dataknobs-config [0.4.3]

#### Changed
- maintenance release

### dataknobs-structures [1.0.14]

#### Changed
- maintenance release

### dataknobs-utils [1.2.17]

#### Changed
- maintenance release

### dataknobs-xization [1.3.13]

#### Changed
- maintenance release

### dataknobs-data [0.6.2]

#### Added
- signal a write under the reserved storage-key name

#### Fixed
- extend the shadowed-id write signal to update/update_batch
- honor the reserved id field on the in-memory ComplexQuery scan path
- consolidate the SQL metadata.-prefix routing into one helper
- fix Query API names in reserved-field notes and examples

### dataknobs-fsm [0.3.1]

#### Changed
- maintenance release

### dataknobs-legacy [0.1.10]

#### Changed
- maintenance release

### dataknobs-llm [0.6.7]

#### Changed
- maintenance release

## Release - 2026-07-18

### dataknobs-common [1.6.1]

#### Added
- added missing_from classmethod; document require_components presence semantics
- added post-construction component injection to StructuredConfigConsumer
- added value-based exception-retry predicate to RetryConfig
- added RetryExecutor.execute_sync; compose allocate over the shared engine

#### Fixed
- hardened RetryExecutor at the result level; validate max_attempts on RetryConfig

### dataknobs-data [0.6.1]

#### Added
- added RetryExecutor.execute_sync; compose allocate over the shared engine

#### Fixed
- hardened RetryExecutor at the result level; validate max_attempts on RetryConfig

### dataknobs-bots [0.9.1]

#### Added
- added post-construction component injection to StructuredConfigConsumer


## Release - 2026-07-15

### dataknobs-common [1.6.0]

#### Added
- unified `Capability.CONDITIONAL_WRITE` compare-and-set advertisement (shared by data + knowledge backends)
- `sweep_stale_test_indices` + `is_elasticsearch_available` / `requires_elasticsearch` test helpers

#### Changed
- ES pytest plugin sweeps stale `test_*` indices at session start (reclaims shard budget)
- service availability probes resolve host Docker-aware

#### Removed
- **Breaking:** removed `Capability.TRANSACTIONAL_METADATA` (superseded by `Capability.CONDITIONAL_WRITE`)

### dataknobs-config [0.4.2]

#### Changed
- maintenance release; refreshed cross-package dependency constraints

### dataknobs-data [0.6.0]

#### Added
- opt-in optimistic concurrency: `get_version()` + `expected_version` compare-and-set on update/upsert/delete across all backends
- atomic `create()` / `create_batch()` raising `DuplicateRecordError` on a colliding id
- `upsert_batch()` write verb; `Operator.STARTS_WITH` prefix predicate; uniform `Filter("id", …)` filterability
- `allocate` / `allocate_sync` monotonic-key allocation; `Migrator` `on_conflict` policy (insert/upsert/skip)

#### Changed
- **Breaking:** `create()` fails closed on a duplicate id instead of silently overwriting
- **Breaking:** Elasticsearch query semantics unified — SQL-wildcard/case-insensitive `LIKE`, full-value `REGEX`, unsupported operators raise instead of matching all
- string operators match only string values (JSON non-string fields no longer coerced)
- multi-kind buffered transactions commit all-or-nothing on transactional backends; migrator rides native bulk verbs

#### Fixed
- async ES honors the full operator set (REGEX/EXISTS/NOT_LIKE/negations) + `ComplexQuery`
- `Filter("id", …)` resolves to the storage key on async S3 / async ES; honest partial-batch failure accounting

### dataknobs-fsm [0.3.0]

#### Removed
- **Breaking:** retired the strategy-based transaction coordinator (`core.transactions`, `TransactionConfig`, `configure_transactions`, `on_transaction_*` hooks) — it drove no real DB atomicity

#### Changed
- `DatabaseResource.commit_batch` rides `upsert_batch`; multi-kind `DatabaseTransaction` commits atomically; `BatchCommit(atomicity="require")` honored on the idempotent-upsert path

#### Security
- bump `click` floor to `>=8.3.3` (CVE-2026-7246)

### dataknobs-llm [0.6.6]

#### Security
- bump `transformers` floor to `>=5.5.0` (CVE-2026-5241); bump `torch` floor to `>=2.13.0` to sweep transitive `setuptools` (CVE-2026-59890)

### dataknobs-structures [1.0.13]

#### Changed
- maintenance release; refreshed cross-package dependency constraints

### dataknobs-utils [1.2.16]

#### Added
- `SimplifiedElasticsearchIndex` atomic create (`op_type="create"` + `ElasticsearchConflictError`) and optimistic-concurrency `if_seq_no` / `if_primary_term` on update/delete

#### Fixed
- percent-encode the document id in ES REST paths (slash/special-char ids round-trip)

#### Security
- bump `nltk` floor to `>=3.10.0` (CVE-2026-54293)

### dataknobs-xization [1.3.12]

#### Security
- bump `nltk` floor to `>=3.10.0` (CVE-2026-54293)

### dataknobs-bots [0.9.0]

#### Changed
- **Breaking:** knowledge backends advertise `Capability.CONDITIONAL_WRITE` (old `TRANSACTIONAL_METADATA` identifier no longer resolves)

### dataknobs [0.1.9]

#### Changed
- maintenance version bump

## Release - 2026-07-07

### dataknobs-common [1.5.2]

#### Added
- added AWS session abstraction (was S3 session)

### dataknobs-utils [1.2.15]

#### Changed
- bumped versions, updated release notes

### dataknobs-xization [1.3.11]

#### Changed
- bumped versions, updated release notes

### dataknobs-data [0.5.5]

#### Changed
- bump transformers floor; acknowledge remaining floor CVEs

### dataknobs-fsm [0.2.5]

#### Changed
- bump transformers floor; acknowledge remaining floor CVEs

### dataknobs-llm [0.6.5]

#### Added
- add Amazon Bedrock LLM + embedding provider

#### Changed
- bumped versions, updated release notes

### dataknobs-bots [0.8.3]

#### Changed
- Renamed S3SessionConfig to AwsSessionConfig and relocated to dataknobs-common


## Release - 2026-06-29

### dataknobs-common [1.5.1]

#### Added
- feat(common): add async->sync bridge for running coroutines from sync code

#### Fixed
- fix(common): address SyncLoopBridge deep-review findings
- fix(common): address SyncLoopBridge deep-review findings
- fix(common): atomic PluginRegistry init + numpy 2.5 mypy stub compat

### dataknobs-config [0.4.1]

#### Changed
- bumped versions, updated release notes

#### Fixed
- fix(common): atomic PluginRegistry init + numpy 2.5 mypy stub compat

### dataknobs-structures [1.0.12]

#### Changed
- bumped versions, updated release notes

### dataknobs-xization [1.3.10]

#### Fixed
- fix(common): atomic PluginRegistry init + numpy 2.5 mypy stub compat

### dataknobs-data [0.5.4]

#### Added
- feat(data,fsm): add AsyncDatabase buffered transaction; make DatabaseTransaction real

#### Fixed
- fix(data,fsm): make BufferedTransaction.is_atomic composition-aware

### dataknobs-fsm [0.2.4]

#### Added
- feat(fsm): wire config-authored builtin/custom function references end-to-end
- feat(fsm): delete the standalone sync execution engine; one async engine
- fix(fsm): release start-state resources on batch/stream runs; add AdvancedFSM lifecycle close
- feat(fsm): run sync FSM APIs on the single async engine via a shared bridge
- feat(fsm): execute push arcs on the async engine


## Release - 2026-06-23

### dataknobs-data [0.5.3]

#### Fixed
- fixed close pool-manager refcount races and partial-connect leaks
- fixed refcount shared pools; close() releases instead of tearing down

### dataknobs-fsm [0.2.3]

#### Changed
- updated dependencies

### dataknobs-llm [0.6.4]

#### Changed
- updated dependencies

### dataknobs-bots [0.8.1]

#### Changed
- updated dependencies


## Release - 2026-06-22

### dataknobs-common [1.5.0]

#### Added
- Add TenantContext protocol + reference implementations for per-tenant scoping
- Add CallbackRegistry with pluggable ordering, error policy, and EventBus fan-out
- Add Capability enum + CapabilityContract/Mixin for declarative feature advertisement
- Add Discriminator and ResourceResolver protocols + reference implementations
- Add vector-store partition resolvers (null, metadata-key, temporal, callable, joining)
- Add ScopeProjector protocol + reference implementations
- Add BackendRegistry protocol unifying Registry and PluginRegistry
- Add rate_limiter_backends / resolver_backends / partition_resolver_backends plugin registries
- Add async factory shims: create_event_bus_async, create_lock_async, create_rate_limiter_async
- Add aiter_sync_in_thread to stream a blocking sync iterator off the event loop
- Add close_if_owned / close_if_owned_sync owned-vs-injected teardown guards
- Add assert_no_blocking() test construct + no_blocking pytest fixture
- Add forwardable_components() to StructuredConfigConsumer for composing strategies
- Extend safe_eval allowlist with frozenset, sum, any, all, reversed

#### Changed
- BREAKING: Wrap backend-factory construction errors in OperationError for create_event_bus / create_lock / create_rate_limiter (catch DataknobsError)
- BREAKING: Convert event_bus_backends and lock_backends to PluginRegistry; use BackendRegistry for isinstance checks (unregister now returns None)
- Enforce the ruff ASYNC lint family

#### Fixed
- Offload PyrateRateLimiter bucket I/O off the event loop for blocking backends
- Release the owned bucket transport on PyrateRateLimiter close() / reset()
- Back off PyrateRateLimiter.acquire() poll interval exponentially under contention
- Block .format() / .format_map() sandbox-escape vector in safe_eval

### dataknobs-data [0.5.2]

#### Added
- Add PgVectorStore.from_components(pool=...) for externally supplied connection pools

#### Changed
- Document async-transport and backing-resource ownership contracts; enforce ruff ASYNC

#### Fixed
- Run async file/vector-store I/O and the aioboto3 session warm off the event loop
- Offload AsyncSQLite / AsyncDuckDB connect directory creation off the loop
- Fix S3 search sort error on falsy sort values such as numeric 0
- Snapshot consistent state in MemoryVectorStore / FaissVectorStore save; handle bare-filename persist_path

### dataknobs-fsm [0.2.2]

#### Changed
- Enforce the ruff ASYNC lint family

#### Fixed
- Run FileProcessor / DatabaseETL pipelines on the async engine; DatabaseETL now executes end-to-end
- Offload FSM file-processing and streaming reads/writes off the event loop
- Accept Path source/sink in AsyncSimpleFSM.process_stream; raise on unsupported compression

#### Security
- Bump langchain floor (llm extra) to >=1.3.9 for GHSA-gr75-jv2w-4656
- Bump aiohttp floor (http extra) to >=3.14.1

### dataknobs-llm [0.6.3]

#### Added
- Add dataknobs_llm.intent module (IntentClassifier protocol, keyword/LLM/composite classifiers, NegationFilter, plugin registry, factories)
- Add CallbackRegistry composition to ExecutionTracker (execution:record fan-out)

#### Fixed
- Offload SqliteEmbeddingCache.initialize directory creation off the event loop

#### Security
- Bump aiohttp floor (ollama/huggingface extras) to >=3.14.1
- Acknowledge accepted torch CVE GHSA-rrmf-rvhw-rf47 (torch.jit.script unused)

### dataknobs-utils [1.2.14]

#### Security
- Acknowledge accepted nltk CVE GHSA-p4gq-832x-fm9v (fixed corpus names only; no caller-controlled input to nltk.data.find())

### dataknobs-xization [1.3.9]

#### Changed
- Enforce the ruff ASYNC lint family

#### Fixed
- Offload LocalDocumentSource.iter_files glob/stat off the event loop
- Offload DirectoryProcessor YAML/CSV conversion and streamed JSON/JSONL ingest off the loop

### dataknobs-bots [0.8.0]

#### Added
- Add intent_confirm: wizard stage primitive + extensible stage-synthesizer registry
- Add clarification_template: stage field for conversation-mode re-renders
- Add turn-lifecycle hooks (LifecycleHooks / WizardHooks on_turn_start / on_turn_end)
- Add StateBridge protocol + reference implementations for named-key state bridging
- Add tenant-scoped chunks to RAGKnowledgeBase and KnowledgeIngestionManager (tenant_id)
- Add per-tenant ingest-state isolation on knowledge backends (TenantContext)
- Add optimistic-concurrency state writes to knowledge backends (expected_version)
- Add knowledge-layer event topics + KnowledgeIngestionManager lifecycle events
- Add KnowledgeKeyKind classification, key_pattern, and subscribe_to_changes for event-driven ingestion
- Add JinjaInputsProjector + declarative wizard stage inputs:
- Add capability advertisement across knowledge bases, managers, and backends
- Add extra_metadata / tenant_id keyword params across RAGKnowledgeBase ingest entry points
- Forward construction collaborators to per-stage wizard sub-strategies (enables composing reasoning: pipeline stages)

#### Changed
- close() across the bot stack tears down only collaborators the holder owns (migration: direct-construction default flipped to leave injected collaborators open)
- Re-platform LifecycleHooks / WizardHooks onto the CallbackRegistry substrate
- Publish knowledge ingestion lifecycle on ingest:domain:start / :end (replaces knowledge:ingestion)
- Run the S3 knowledge backend on aioboto3 and offload file-backend I/O off the loop
- Enforce the ruff ASYNC lint family

#### Fixed
- Fix cross-tenant chunk_id UPSERT collision in shared RAGKnowledgeBase instances
- Fix cross-tenant filter-based deletion in KnowledgeIngestionManager
- Offload wizard config and knowledge-base tool disk I/O off the event loop

#### Security
- Reject path-traversal config/resource names in SaveConfigTool and the KB wizard tools
- Bump starlette floor (server extra) to >=1.3.1 for GHSA-82w8-qh3p-5jfq and related sweeps


## Release - 2026-06-08

### dataknobs-data [0.5.1]

#### Changed
- Updated dependencies

### dataknobs-fsm [0.2.1]

#### Changed
- Updated dependencies

### dataknobs-bots [0.7.3]

#### Added
- Add public accessors for reasoning-strategy state on DynaBot

#### Changed
- Updated dependencies


## Release - 2026-06-06

### dataknobs-llm [0.6.2]

#### Added
- Add seed-metadata API on ConversationManager

#### Fixed
- Fix ContextPersister.persist and document the seed-metadata API

### dataknobs-bots [0.7.2]

#### Added
- Add prompt_envelope on DynaBotConfig; markdown default

#### Fixed
- Fix ContextPersister.persist and document the seed-metadata API


## Release - 2026-06-02

### dataknobs-xization [1.3.8]

#### Changed
- bumped versions, updated release notes

### dataknobs-legacy [0.1.7]

#### Changed
- bumped versions, updated release notes

### dataknobs-llm [0.6.1]

#### Added
- add citation-carryover redaction for buffer memory + conversation middleware

#### Changed
- Relocate HistoryRedaction to dataknobs-llm and generalize the helper
- chore: bump torch floor for PYSEC-2026-139 and cap aiohttp for aioresponses

### dataknobs-bots [0.7.1]

#### Added
- add citation-carryover redaction for buffer memory + conversation middleware

#### Changed
- Extend read-time history redaction to Summary/Vector memory + split middleware helper
- Rewrite Unreleased CHANGELOG entries for end-state-only style
- chore: bump torch floor for PYSEC-2026-139 and cap aiohttp for aioresponses


## Release - 2026-05-26

### dataknobs-common [1.4.0]

#### Added
- Add StructuredConfig.validate() for polymorphic config sections
- Add automatic secret redaction to StructuredConfig repr
- Add collaborator injection and async dispatch to StructuredConfigConsumer
- add StructuredConfig meta-abstraction + retrofit event-bus backends
- add structured event-bus configs + cross-registry parity guards

#### Changed
- Adopt polymorphic-section validation for bots memory/knowledge_base
- Coerce Enum fields in StructuredConfig.from_dict; add to_json_dict + jsonify
- Migrate PostgresAdvisoryLock to StructuredConfigConsumer
- Support async-canonical from_config for structured-config consumers
- Make collaborator delivery signature-aware; clarify from_components errors
- generalize StructuredConfig construction-lifecycle & composition

### dataknobs-config [0.4.0]

#### Added
- add StructuredConfig meta-abstraction + retrofit event-bus backends

#### Changed
- generalize StructuredConfig construction-lifecycle & composition

### dataknobs-structures [1.0.11]

#### Changed
- bumped versions, updated release notes

### dataknobs-utils [1.2.13]

#### Changed
- Migrate all dataknobs-data backends to typed StructuredConfig
- bumped versions, updated release notes

### dataknobs-data [0.5.0]

#### Added
- Add StructuredConfig.validate() for polymorphic config sections
- Add automatic secret redaction to StructuredConfig repr
- add structured event-bus configs + cross-registry parity guards

#### Changed
- Bump dependency floors for floor-resolve CVE findings
- Migrate cross-cutting leaf configs to StructuredConfig
- Migrate vector stores to StructuredConfig + formalize empty-list filter contract
- Migrate all dataknobs-data backends to typed StructuredConfig

### dataknobs-fsm [0.2.0]

#### Added
- Add fail-closed regression guard for StorageConfig enum-keyed JSON round-trip

#### Changed
- Adopt StructuredConfigConsumer in FSM pattern/runtime consumers
- Migrate FSM resources/IO/storage/streaming/functions configs to StructuredConfig
- Migrate FSM patterns-family configs to StructuredConfig
- Bump dependency floors for floor-resolve CVE findings

### dataknobs-llm [0.6.0]

#### Added
- Add embedding-section typed config + validation (reuse LLMConfig)
- Add LLMConfig field-coverage drift guard for LLM providers
- add structured event-bus configs + cross-registry parity guards

#### Changed
- Migrate LLMConfig to frozen StructuredConfig + llm-section validation
- Bump dependency floors for floor-resolve CVE findings

### dataknobs-bots [0.7.0]

#### Added
- Add embedding-section typed config + validation (reuse LLMConfig)
- Add StructuredConfig.validate() for polymorphic config sections
- Add automatic secret redaction to StructuredConfig repr
- Add builder↔ctor drift guards for memory backend builders
- Add pluggable backend registries for memory, knowledge, and grounded sources

#### Changed
- Adopt StructuredConfigConsumer mixin in the wizard reasoning strategy
- Adopt StructuredConfigConsumer mixin in non-wizard reasoning strategies
- Validate the reasoning config section in DynaBotConfig
- Migrate wizard reasoning config family to StructuredConfig
- Migrate core reasoning-strategy configs to StructuredConfig
- Migrate LLMConfig to frozen StructuredConfig + llm-section validation
- Convert config-validation skip-sentinel test teardown to a fixture
- Adopt polymorphic-section validation for bots memory/knowledge_base
- Bump dependency floors for floor-resolve CVE findings
- Descend StructuredConfig repr redaction into raw mapping/list fields
- Migrate vector stores to StructuredConfig + formalize empty-list filter contract
- Adopt StructuredConfigConsumer in memory and knowledge subsystem classes
- Adopt typed DynaBotConfig and structured-config lifecycle in DynaBot
- Migrate bots config/ dataclasses to StructuredConfig
- bumped versions, updated release notes


## Release - 2026-05-20

### dataknobs-common [1.3.14]

#### Added
- add ensure_localstack_s3_bucket helper and pytest11 fixture plugin
- add SqsEventBus single-topic bridge mode

#### Fixed
- expose LocalStack endpoint resolver

### dataknobs-xization [1.3.7]

#### Changed
- bumped versions, updated release notes

### dataknobs-data [0.4.20]

#### Added
- add ensure_localstack_s3_bucket helper and pytest11 fixture plugin

#### Changed
- bumped versions, updated release notes

### dataknobs-fsm [0.1.21]

#### Changed
- bumped versions, updated release notes

### dataknobs-legacy [0.1.6]

#### Changed
- bumped versions, updated release notes

### dataknobs-llm [0.5.14]

#### Changed
- bumped versions, updated release notes


## Release - 2026-05-19

### dataknobs-bots [0.6.22]

#### Added
- added an ingestion manager resolver for the ingest orchestrator

### dataknobs-structures [1.0.10]

#### Changed
- bumped versions, updated release notes

### dataknobs-utils [1.2.12]

#### Changed
- bumped versions, updated release notes


## Release - 2026-05-18

### dataknobs-common [1.3.13]

#### Added
- Add PostgresAdvisoryLock cross-replica DistributedLock backend
- feat(common): add internal run_supervised_loop helper for event-bus listeners
- feat(common): extract compute_backoff_delay as a public pure function
- feat(common): add SqsEventBus backend + optional-dependency extras
- Add dataknobs_common.locks distributed lock abstraction (Item 128 Phase 1)
- feat(common): registry-extensible event bus backends (Item 127 Phase 1)

#### Changed
- docs(common): document event-bus connection resilience; CHANGELOG
- common: reword plan-tracking refs (no behavior change)
- common: reword internal tracking labels (no behavior change)
- common: shared make_pgvector_test_table fixture + gated orphan sweep (Item 129 Changes A+B)

#### Fixed
- fix(common): pace Postgres LISTEN watchdog; add is_listening; doc compute_backoff_delay
- fix(common): PostgresEventBus reconnects a dropped LISTEN connection
- fix(common): RedisEventBus re-establishes pub/sub on connection loss
- refactor(common): SqsEventBus on shared supervised loop; fix topic starvation
- fix(common): sqs factory raises clean ValueError; guard aioboto3-free import
- common: shared make_pgvector_test_table fixture + gated orphan sweep (Item 129 Changes A+B)

### dataknobs-config [0.3.14]

#### Changed
- config: reword internal tracking labels (no behavior change)

### dataknobs-xization [1.3.6]

#### Added
- Add per-file delta ingestion to KnowledgeIngestionManager

#### Changed
- fsm/llm/xization: reword plan-phase refs (no behavior change)
- xization: reword internal tracking labels (no behavior change)
- bumped versions, updated release notes

### dataknobs-data [0.4.19]

#### Added
- feat(data): expose AsyncS3Database.region for sync/async parity
- Address PR5A triage findings: label leakage, test fake, lazy logging
- Item 131: add cross-backend metadata-aliasing conformance test
- PgVectorStore init-time dimension-mismatch guard (Item 129 Change C)

#### Fixed
- encode non-scalar Chroma metadata to stop cross-collection corruption
- Make ChromaVectorStore compatible with chromadb 1.x
- reword plan-phase refs (no behavior change)
- reword PR#/review# refs (no behavior change)
- reword internal tracking labels (no behavior change)
- Item 130: replace IVF make_direct_map with raw-vector side-car (supersedes 887430f5)
- Item 130: fix FaissVectorStore.get_vectors() for IVF index types
- Item 132: neutralize PgVectorStore default schema (edubot -> public)
- Item 133: remove pre-existing internal tracking labels from committed source
- Address PR #317 review: fix TOMBSTONE additive-delta data loss
- Redesign TOMBSTONE re-ingest: crash-safe generation swap

### dataknobs-fsm [0.1.20]

#### Fixed
- Make ChromaVectorStore compatible with chromadb 1.x
- reword plan-phase refs (no behavior change)
- reword internal tracking labels (no behavior change)

### dataknobs-llm [0.5.13]

#### Fixed
- reword plan-phase refs (no behavior change)
- reword internal tracking labels (no behavior change)

### dataknobs-bots [0.6.21]

#### Added
- Add PostgresAdvisoryLock cross-replica DistributedLock backend
- PR7 (125/126 Phase 4): embedder rate-limit seam + config-driven orchestrator lock
- Add native per-version snapshots and orchestrator dispatch matrix
- Add per-file delta ingestion to KnowledgeIngestionManager
- Add dataknobs_common.locks distributed lock abstraction (Item 128 Phase 1)

#### Fixed
- Extract shared requires_real_postgres mark; doc/changelog accuracy
- updated CVE dependency floors and ran quality checks
- reword plan-tracking refs (no behavior change)
- reword internal tracking labels (no behavior change)
- Unify the knowledge-backend version model (Items 125/126 Phase 0)


## Release - 2026-05-13

### dataknobs-config [0.3.13]

#### Changed
- bumped pyyaml and psycopg2-binary floors to installable versions

### dataknobs-structures [1.0.9]

#### Changed
- bumped versions, updated release notes

### dataknobs-utils [1.2.11]

#### Changed
- bumped pyyaml and psycopg2-binary floors to installable versions

### dataknobs-data [0.4.18]

#### Added
- added and wired in the key record store abstraction for consistent (and corrected) handling across all sites

#### Fixed
- fixed elasticsearch testing table leaks

### dataknobs-fsm [0.1.19]

#### Added
- added and wired in the key record store abstraction for consistent (and corrected) handling across all sites

#### Changed
- bumped floors to clear CVE findings on update-dependencies branch
- bumped pyyaml and psycopg2-binary floors to installable versions

### dataknobs-llm [0.5.12]

#### Changed
- bumped floors to clear CVE findings on update-dependencies branch
- bumped pyyaml and psycopg2-binary floors to installable versions


## Release - 2026-05-09

### dataknobs-common [1.3.12]

#### Fixed
- fixed destructive vector database metadata merge order patterns
- shared yaml/json loader helper

### dataknobs-config [0.3.12]

#### Added
- added floor cve sweep to dependency updates, and performed the sweep

#### Changed
- added floor cve sweep to dependency updates, and performed the sweep

#### Fixed
- shared yaml/json loader helper
- canonicalized env-var substitution
- fixed environment config env var substitution

### dataknobs-utils [1.2.10]

#### Added
- added floor cve sweep to dependency updates, and performed the sweep

#### Changed
- bumped versions, updated release notes

### dataknobs-xization [1.3.5]

#### Added
- added floor cve sweep to dependency updates, and performed the sweep

#### Fixed
- fixed destructive vector database metadata merge order patterns
- shared yaml/json loader helper

### dataknobs-data [0.4.17]

#### Added
- added floor cve sweep to dependency updates, and performed the sweep

#### Changed
- bumped versions, updated release notes

#### Fixed
- fixed destructive vector database metadata merge order patterns
- PgVectorStore postgres identifier validation + harmonized validate_database_name exception
- async postgres row to record id asymmetry

### dataknobs-fsm [0.1.18]

#### Added

- added floor cve sweep to dependency updates, and performed the sweep
- shared yaml/json loader helper

#### Fixed
- rewrote fsm history CLI commands against real BaseHistoryStorage API
- fixed FSM unified storage backend selection and config plumbing
- migrated FSM env-var resolver

### dataknobs-legacy [0.1.4]

#### Added

- added floor cve sweep to dependency updates, and performed the sweep

#### Fixed
- bumped versions, updated release notes

### dataknobs-llm [0.5.11]

#### Added
- added floor cve sweep to dependency updates, and performed the sweep
- shared yaml/json loader helper

### dataknobs-bots [0.6.18]

#### Added
- added floor cve sweep to dependency updates, and performed the sweep
- added peek to registry backend for no-touch access
- shared yaml/json loader helper

#### Fixed
- fixed destructive vector database metadata merge order patterns
- ensured ingestion result completed at on skip


## Release - 2026-05-06

### dataknobs-config [0.3.11]

#### Changed
- bumped versions, updated release notes

### dataknobs-structures [1.0.8]

#### Changed
- bumped versions, updated release notes

### dataknobs-xization [1.3.4]

#### Changed
- bumped versions, updated release notes

### dataknobs-fsm [0.1.17]

#### Changed
- bumped versions, updated release notes

### dataknobs-llm [0.5.10]

#### Added
- added parallel llm executor cancellation

### dataknobs-bots [0.6.18]

#### Changed
- bumped versions, updated release notes


## Release - 2026-04-29

### dataknobs-common [1.3.11]

#### Fixed
- collected shared integration test fixtures
- fixed dropped metadata bug

### dataknobs-utils [1.2.9]

#### Fixed
- sql injection hardening for non-identifier field names and DataFrame upload
- fixed safe sql quoting behavior for production code paths

### dataknobs-data [0.4.16]

#### Fixed
- sql injection hardening for non-identifier field names and DataFrame upload
- fixed safe sql quoting behavior for production code paths
- fixed to parameterize auto create table
- fixed to remove f-string SQL strings in tests
- collected shared integration test fixtures
- fixed dropped metadata bug

### dataknobs-legacy [0.1.3]

#### Changed
- bumped versions, updated release notes

### dataknobs-llm [0.5.9]

#### Fixed
- collected shared integration test fixtures
- fixed dropped metadata bug


## Release - 2026-04-23

### dataknobs-common [1.3.10]

#### Fixed
- fixed post-PR code review issues

### dataknobs-config [0.3.10]

#### Changed
- bumped versions, updated release notes

### dataknobs-structures [1.0.7]

#### Changed
- bumped versions, updated release notes

### dataknobs-utils [1.2.8]

### dataknobs-xization [1.3.3]

#### Changed
- bumped versions, updated release notes

### dataknobs-data [0.4.15]

#### Added
- added vector store list membership filters and fixed filter translation type safety
- added updated_at column to pgvector and memory vector stores (deferred faiss and chroma)
- added s3 region fallback
- added filter passthrough and identity callables to VectorKnowledgeSource

#### Changed
- added updated_at column to pgvector and memory vector stores (deferred faiss and chroma)
- bumped versions, updated release notes

#### Fixed
- added vector store list membership filters and fixed filter translation type safety
- fixed post-PR code review issues
- fixed post-PR code review issues

### dataknobs-fsm [0.1.16]

#### Changed
- bumped versions, updated release notes

### dataknobs-llm [0.5.8]

#### Added
- added opt-in persistable middleware metadata


## Release - 2026-04-15

### dataknobs-utils [1.2.7]

#### Added
- added extraction value expansion

#### Fixed
- fixed for string-aware brace matching

### dataknobs-llm [0.5.7]

#### Added
- added TemplateSyntax enum and conversation utilities

#### Fixed
- enhanced prompt framework

### dataknobs-bots [0.6.16]

#### Fixed
- centralized configurable prompts
- fixed extractor grounding to use extraction value expansion (opt-in per-field)
- fixed extraction type mismatch
- streamlined design and consolidated code
- fixed extraction grounding to include first-write bypass


## Release - 2026-04-13

### dataknobs-bots [0.6.15]

#### Fixed
- fixed inconsistency bugs by refactoring monolithic confirmation logic
- fixed extraction confirmation gap
- fixed render_count bug on subflow push (w/test-first exposure)
- refactored to consolidate render_count tracking and stream/no-stream generate_stage_response code paths


## Release - 2026-04-11

### dataknobs-bots [0.6.14]

#### Changed
- redesigned and refactored wizard.py

#### Fixed
- fixed several bugs in wizard.py


## Release - 2026-04-07

### dataknobs-bots [0.6.13]

#### Added
- refactor: added StageSchema for consolidating logic and migrated code to use it

#### Fixed
- bug fixes: fixed conversation-mode template loop and confidence gate


## Release - 2026-04-06

### dataknobs-bots [0.6.12]

#### Fixed
- fixed hybrid reasoning strategy design defects


## Release - 2026-04-06

### dataknobs-xization [1.3.2]

#### Added
- added chunker abstraction
- added chunk character position tracking and chunk transform pipeline

### dataknobs-bots [0.6.11]

#### Added
- updated to leverage the plugable chunker abstraction through configs


## Release - 2026-04-04

### dataknobs-bots [0.6.10]

#### Added
- added modular reasoning strategy registry with plugin-based discovery (PR #233)
- added per-stage reasoning strategy injection for wizard states (PR #235)

#### Fixed
- fixed Jinja2 template rendering to use sandboxed environment across all template sites (PR #236)

### dataknobs-common [1.3.9]

#### Added
- generalized PluginRegistry with lazy loading, dependency ordering, and config-driven instantiation (PR #234)

### dataknobs-data [0.4.14]

#### Fixed
- refactored backend and vector store registries to leverage enhanced common PluginRegistry (PR #234)
- fixed deprecation warnings from Python 3.12 upgrade

### dataknobs-fsm [0.1.15]

#### Fixed
- fixed shared database collisions when history and step storage use the same backing database (PR #237)

### dataknobs-llm [0.5.6]

#### Fixed
- refactored LLM provider registry to leverage enhanced common PluginRegistry (PR #234)
- fixed deprecation warnings from Python 3.12 upgrade

### dataknobs-utils [1.2.6]

#### Fixed
- modernized type hints to Python 3.12 syntax (PR #232)

### dataknobs-config [0.3.9]

#### Changed
- bumped minimum Python version to 3.12

### dataknobs-structures [1.0.6]

#### Changed
- bumped minimum Python version to 3.12

### dataknobs-xization [1.3.1]

#### Changed
- bumped minimum Python version to 3.12

### Infrastructure

- bumped minimum Python version to 3.12 across all packages (PR #232)


## Release - 2026-04-03

### dataknobs-common [1.3.8]

#### Added
- added backing for redis

#### Fixed
- more bug fixes


## Release - 2026-04-01

### dataknobs-common [1.3.7]

#### Fixed
- bug fixes

### dataknobs-llm [0.5.5]

#### Added
- migrated create_embedding_provider to the llm package from bots

### dataknobs-bots [0.6.9]

#### Added
- added routing_transforms

#### Fixed
- redesigned wizard generate to separate business logic and extraction from presentation
- fixed load_markdown_text api to be public
- migrated create_embedding_provider to the llm package


## Release - 2026-03-31

### dataknobs-common [1.3.6]

#### Added
- added expression engine abstraction

### dataknobs-bots [0.6.8]

#### Added
- added hybrid reasoning mode, composing grounded and react reasoning
- migrated expression impls to common package's engine abstraction
- added wizard transforms for conditional/logical, collections, regex, and general-purpose


## Release - 2026-03-30

### dataknobs-bots [0.6.7]

#### Added
- added grounded reasoning strategy with configurable search result
  synthesis and deterministic retrieval (PR #216)
- added standalone extraction grounding utility for reuse across
  reasoning strategies (PR #216)
- added per-stage first-render confirmation control for wizard
  flows (PR #215)
- added composite memory fallback and embedding provider factory
  improvements (PR #211)
- added wizard loader validation for stage configuration (PR #211)

#### Fixed
- fixed tool/middleware error propagation and timeout handling in
  turn lifecycle (PR #213)
- fixed process hanging and error swallowing during bot
  creation (PR #214)
- fixed metadata dropping in grounded reasoning pipeline (PR #216)
- fixed thinking mode interference with extraction (PR #216)

### dataknobs-llm [0.5.4]

#### Added
- added extraction grounding utility for validating extracted
  values against field schemas (PR #216)
- added retrieval intent types for structured source
  queries (PR #216)

#### Fixed
- fixed Anthropic messages bug by standardizing LLM adapter pattern
  across all providers (PR #212)
- fixed Ollama provider model matching to be strict (PR #211)

### dataknobs-data [0.4.13]

#### Added
- added grounded source abstraction with database, topic index,
  and cluster index implementations (PR #216)
- added cross-source normalization and result processing
  utilities (PR #216)

#### Fixed
- fixed `LIKE`/`NOT_LIKE` filter operators to be
  case-insensitive (PR #216)

### dataknobs-fsm [0.1.14]

#### Fixed
- fixed async HTTP provider session cleanup to drain SSL transport
  callbacks before event loop shutdown (PR #214)

### dataknobs-config [0.3.8]

#### Fixed
- fixed `substitute_env_vars` to also substitute environment
  variables in dictionary keys

### dataknobs-xization [1.3.0]

#### Added
- added HTML-to-markdown converter with structure-preserving
  table and list handling (PR #209)

#### Fixed
- replaced `chunk_overlap` with priority-based boundary splitting
  in `MarkdownChunker` (paragraph, sentence, word) (PR #220)

### dataknobs-common [1.3.5]

#### Fixed
- testing utility with markdown chunk_overlap parameter removal (PR #220)

### Infrastructure / CI

- added vulnerability auditing with `osv-scanner` (PR #210)
- updated `nltk` and `torch` for CVE remediation (PR #210)
- simplified dependency-update workflow to Python-only (PR #218)
- bumped GitHub Actions in the `github-actions` group (PR #219)


## Release - 2026-03-23

### dataknobs-bots [0.6.6]

#### Added
- added `BotTestHarness` and `WizardConfigBuilder` testing utilities for
  standardized bot test setup (PR #184)
- added `TurnState` per-turn cross-middleware communication and bridged
  LLM + state middleware (PR #186)
- added `from_config` direct injection capability for providers and
  middleware (PR #186)
- added wizard extraction field grounding to validate extracted values
  against field schemas (PR #181)
- added extraction scope escalation strategy for multi-field extraction
  retries (PR #183)
- added wizard extractor field derivations for computed/dependent
  fields (PR #187)
- added enum-based extraction normalization in hints framework (PR #188)
- added extraction recovery pipeline for retrying failed
  extractions (PR #189)
- added custom merge filter protocol for wizard data merging (PR #190)
- added boolean extraction recovery with negation handling (PR #192)
- added security hardening for `context_transform` and summary memory
  injection resistance (PR #186)

#### Fixed
- fixed auto_advance and override logic and landing stage extraction
  from transition messages (PR #175)
- fixed `skip_extraction` lifecycle and stale `_message` injection in
  wizard reasoning (PR #176)
- fixed `store_trace` and `verbose` forwarding through ReAct wizard
  reasoning (PR #177)
- fixed wizard extraction from polluted prompts by managing raw user
  content (PR #178)
- fixed partial wizard data accumulation across multi-turn
  extraction (PRs #179, #180)
- fixed strategy tools gap — reject non-enum values in tool
  registration (PR #191)
- unified hook migration, deprecating legacy hooks (PR #186)

### dataknobs-llm [0.5.3]

#### Added
- added `turn_data` transient state on `ConversationState` for per-turn
  cross-middleware communication (PR #186)
- added `turn_data` bridging into `ToolExecutionContext` so tools can
  access per-turn plugin data (PR #186)
- added `strict_tools` mode on `EchoProvider` to catch missing tool
  definitions in tests (PR #191)
- added `ConfigurableExtractor` and `scripted_schema_extractor` testing
  utilities (PR #184)

#### Fixed
- improved extraction prompts — explicit omission rules, boolean
  negation handling, better error messages (PR #178)

### dataknobs-fsm [0.1.13]

#### Fixed
- fixed `InMemoryStorage` to use separate databases for history and step
  records, avoiding namespace collisions (PR #185)
- added explicit `owns_databases` parameter on `UnifiedDatabaseStorage`
  for ownership control of injected databases (PR #185)

### dataknobs-config [0.3.7]

#### Fixed
- fixed `substitute_env_vars` to use `os.path.expanduser()` instead of
  `Path.expanduser()`, preventing URL corruption (collapsing `://` to
  `:/`) (PR #185)

### dataknobs-data [0.4.12]

#### Fixed
- fixed PgVectorStore `add_vectors` to upsert all columns (content,
  domain_id, document_id, chunk_index) on ID conflict, preserving
  `created_at` timestamp (PR #185)

### dataknobs-config [0.3.7]

#### Fixed
- miscellaneous bug fixes

### dataknobs-xization [1.2.6]

#### Fixed
- quality review fixes

### Infrastructure / CI

- pinned all GitHub Actions to SHAs for supply chain security (PR #193)
- added Dependabot configuration for automated action updates (PR #193)
- bumped `peter-evans/create-pull-request`, `actions/upload-artifact`,
  `actions/upload-pages-artifact` (PR #204)
- added workflow syntax and pinned SHA validation checks (PR #200)
- updated dependency update workflow to wrap Dependabot PRs in addition
  to the Monday morning schedule (PR #207)


## Release - 2026-03-16

### dataknobs-llm [0.5.2]

#### Added
- added embedding provider factory support for config-driven embedding
provider creation
- added caching embedding provider with pluggable backends (memory, SQLite)
- added provider visibility — summary memory can expose its LLM provider for
  registration

#### Fixed
- fixed SQL dot-notation queries in storage backends
- fixed error handling consistency across chat implementations

### dataknobs-bots [0.6.5]

#### Added
- added pluggable conversation storage via config (`storage_class` key)
- added public wizard advance API for non-conversational wizard progression
- added provider registry on DynaBot for enumerating and managing all
  LLM/embedding providers
- added composite memory strategy combining multiple memory backends
- added embedding provider factory support in memory and knowledge base config

#### Fixed
- fixed error handling consistency across chat and stream_chat

### dataknobs-fsm [0.1.12]

#### Added
- added storage injection — FSM storage backends can be provided externally 
  instead of created internally
- added metadata filtering in query_histories() with dot-notation support

#### Fixed
- refactored AdvancedFSM for shared sync/async execution core, eliminating
  code duplication

### dataknobs-data [0.4.11]

#### Added
- added Postgres database auto-create — databases are created automatically if
  they don't exist

#### Fixed
- fixed SQL dot-notation queries for nested field access in filters


## Release - 2026-03-10

### dataknobs-bots [0.6.4]

#### Added
- added post-stream middleware hook

#### Fixed
- fixed flow to enable wizard message mode behavior (skip states w/out requiring a user response)
- fixed deictic resolution bug
- fixed bug in wizard undo fsm state restoration
- fixed middleware bugs


## Release - 2026-03-09

### dataknobs-utils [1.2.5]

#### Fixed
- resiliency fix for transient elasticsearch errors

### dataknobs-bots [0.6.3]

#### Fixed
- fixed skip navigation and config casing bugs


## Release - 2026-03-06

### dataknobs-common [1.3.3]

#### Fixed
- fixed bugs, including 1 security injection risk

### dataknobs-bots [0.6.2]

#### Added
- Added conversation undo/rewind capability

## Release - 2026-03-05

### dataknobs-common [1.3.2]

#### Added
- added json safety functions and aids for serialization strictness

### dataknobs-config [0.3.6]

#### Fixed
- fixed passing capabilities data through config

### dataknobs-data [0.4.10]

#### Fixed
- fixed async elasticsearch database to override count() for filtered queries

### dataknobs-fsm [0.1.11]

#### Fixed
- improved transition control and data/context management

### dataknobs-llm [0.5.1]

#### Added
- added call tracker utility
- added thinking mode detection
- added LLM capture/replay testing harness support

#### Fixed
- improved conversation management and storage
- improved parallel execution configuration
- fixed provider functionality gaps
- fixed llm message serialization

### dataknobs-bots [0.6.1]

#### Added
- added wizard turn context for separating transient from persistent data
- added greeting for non-wizard bots
- added multi-llm capability validation (e.g., extractor -vs- main llm)
- added bots capture/replay testing utilities

#### Fixed
- fixed fsm context management
- fixed reasoning strategy lifecycle and streaming contracts
- fixed greet initial context


## Release - 2026-03-03

### dataknobs-llm [0.5.0]

#### Added
- added persistence of system prompt overrides to metadata
- added name param to add_message for tool result messages
- added 'tool', 'assistant', and 'function' role support
- added tool_calls to LLMMessage
- added conversation export_to_dict
- added accessor for collecting all conversation nodes

#### Fixed
- fixed chat -vs- chat stream code divergence
- fixed provider tool usage bugs
- fixed to deep-copy tc.parameters in metadata capture to prevent aliasing
- fixed conversation storage bugs

### dataknobs-bots [0.6.0]

#### Added
- added artifact bank abstractions with tools
- added restart_wizard tool
- added wizard artifact catalog lifecycle tools

#### Fixed
- fixed wizard and react reasoning flow, context injection, tools, and bugs
- fixed bugs and sync/async divergences
- fixed conversation metadata update timing
- fix to refresh system prompt on data change through tools


## Release - 2026-02-26

### dataknobs-config [0.3.5]

#### Added
- added validation of $requires against capabilities metadata

### dataknobs-data [0.4.9]

#### Fixed
- resource management fixes

### dataknobs-llm [0.4.0]

#### Added
- improved storage and retrieval for visibility (w/ bots)
- broadened conversation search capabilities
- added llm resource specs and layered enforcement strategies (w/ bots, config)
- added delete conversations by filter
- added metadata accessor
- added a conversation branching helper method
- updated tool-using strategy across llm providers, including deprecation

#### Fixed
- fixes to inject system context variables, including current_date (including performing template rendering without rag — old bug)
- fixed gaps in persisting conversation metadata (w/ bots)
- fixed conversation_id initialization, eliminating wasteful conversation root node (w/ bots)
- fixed to allow injected capabilities
- fixed resource management

### dataknobs-bots [0.5.0]

#### Added
- improved storage and retrieval for visibility (w/ llm)
- added llm resource specs and layered enforcement strategies (w/ llm, config)
- added per-message wizard state snapshots, config validation warnings, and debug logging; updated documentation
- added bot greeting
- added configurable wizard navigation
- added memory bank abstraction

#### Fixed
- fixed dynabot stream_chat to return all information, not just the text — BREAKING CHANGE in return value
- fixed resource leaks (multiple instances)
- fixed gaps in persisting conversation metadata (w/ llm)
- fixed to detect and break duplicate tool calls; fix post-break logic
- fixed resource cleanup bugs (w/ data)
- fixed conversation_id initialization, eliminating wasteful conversation root node (w/ llm)
- fixed wizard reasoning vs conversation manager interface disconnect
- fixed to centralize wizard metadata (across all wizard modes)
- refactored tests to remove bug-obscuring mocks (WizardTestManager)
- fixed state counting bugs using centralized code
- fixed stream_chat's defects/divergence from chat
- fixed conversation tree to properly build branches


## Release - 2026-02-21

### dataknobs-bots [0.4.8]

#### Added
- enhanced artifact registry to support content field filtering
- conversational intent detection for wizard state transition
- an artifact corpus abstraction
- wizard transform helpers for corpus operations
- a generic rate limiter

#### Fixed
- fixed serialization bugs
- fixed async deficiencies
- fixed wizard initialization from config
- fixed wizard state tracking and flow

### dataknobs-common [1.3.1]

#### Added
- a generic rate limiter

#### Fixed
- fixed serialization bugs

### dataknobs-data [0.4.8]

#### Added
- a dedup checker utility

### dataknobs-fsm [0.1.10]

#### Changed
- updated documentation
- miscellaneous fixes and small enhancements

#### Fixed
- refactored to leverage the common package's generic rate limiter
- fix to pass function reference params to transform functions

### dataknobs-llm [0.3.6]

#### Added
- a parallel llm executor

#### Fixed
- fixed disconnected rate limit checking
- refactored to leverage the common package's generic rate limiter
- fixed incomplete async fsm integration layer


## Release - 2026-02-17

### dataknobs-bots [0.4.7]

#### Added
- added conversational intent detection for wizard state transitions


## Release - 2026-02-16

### dataknobs-bots [0.4.6]

#### Added
- added a summary memory option
- added deterministic code generators to be used (and eventually created) by bots
- added artifact provenance and rubric evaluation
- added rubrics extraction


## Release - 2026-02-14

### dataknobs-utils [1.2.4]

#### Fixed
- fixed transitive dependencies

### dataknobs-xization [1.2.5]

#### Fixed
- fixed transitive dependencies

### dataknobs-data [0.4.7]

#### Fixed
- fixed intermittent test failures
- fixed transitive dependencies

### dataknobs-bots [0.4.5]

#### Fixed
- fixed transitive dependencies


## Release - 2026-02-11

### dataknobs-common [1.3.0]

#### Added
- added standalone transition validation functionality in common for general use
- promoted configurable retry logic utilities from fsm to common for general use

### dataknobs-fsm [0.1.9]

#### Fixed
- promoted configurable retry logic utilities from fsm to common for general use

### dataknobs-llm [0.3.5]

#### Fixed
- fixed to properly handle kwargs

### dataknobs-bots [0.4.4]

#### Added
- configbot toolkit

#### Changed
- enhanced tool dependency resolution


## Release - 2026-02-09

### dataknobs-data [0.4.6]

#### Fixed
- linting errors

### dataknobs-fsm [0.1.8]

#### Fixed
- fixed faulty divergent path bug

### dataknobs-bots [0.4.3]

#### Added
- lm context generation and transition data derivation features


## Release - 2026-02-06

### dataknobs-bots [0.4.2]

#### Added
- wizard subflow support
- templated wizard responses
- stage label support
- per-stage `extraction_scope` override
- schema-aware data normalization

#### Fixed
- consistent wizard metadata on all response paths
- wizard state reset on restart
- settings injection from wizard config
- template response persistence through serialization

### dataknobs-fsm [0.1.7]

#### Added
- subflow engine support
- multi-transform arc execution

#### Fixed
- tuple truthiness handling in condition evaluation
- exec() scope bug
- subflow network stack popping on completion

### dataknobs-llm [0.3.4]

#### Fixed
- floating point precision in schema extraction numeric fields


## Release - 2026-01-29

### dataknobs-llm [0.3.3]

#### Added
- added assumption tracking in SchemaExtractor

### dataknobs-bots [0.4.1]

#### Added
- added artifacts, reviews, task injection, focus guards, and config versioning enhancements


## Release - 2026-01-28

### dataknobs-config [0.3.4]

#### Added
- added template variable substitution utility

#### Changed
- updated documentation

### dataknobs-utils [1.2.3]

#### Fixed
- fixed ruff errors

### dataknobs-llm [0.3.2]

#### Added
- added testing utilities
- adding missing close methods

### dataknobs-bots [0.4.0]

#### Added
- add ReAct reasoning to wizard reasoning
- strip schema defaults, and add skip-default handling
- adds for auto-ingestion
- adding missing close methods

#### Changed
- updated documentation
- improved hardcoded/default prompt

#### Fixed
- fixed ruff errors


## Release - 2026-01-23

### dataknobs-common [1.2.1]

#### Fixed
- Tightened dependencies

### dataknobs-xization [1.2.4]

#### Fixed
- Tightened dependencies

### dataknobs-fsm [0.1.6]

#### Added
- Added observability functionality

### dataknobs-llm [0.3.1]

#### Added
- Observability functionality
- Context injection into tools

#### Fixed
- Fixed missing optional dependencies

### dataknobs-bots [0.3.1]

#### Added
- Observability functionality
- Custom function resolution


## Release - 2026-01-14

### dataknobs-bots [0.3.0]

#### Added
- Dynamic Registration
  - DataKnobsRegistryAdapter for pluggable config storage
  - CachingRegistryManager with TTL and event invalidation
  - Hot-reload infrastructure (HotReloadManager, RegistryPoller)
  - HTTPRegistryBackend for REST API config sources
  - Knowledge storage backends (InMemory, File, S3)
  - KnowledgeIngestionManager for file→vector ingestion
- Wizard Reasoning
  - Wizard Reasoning Strategy for FSM-backed guided conversational flows
  - WizardFSM - Thin wrapper around AdvancedFSM with wizard-specific conveniences (navigation, stage metadata, state serialization)
  - WizardConfigLoader - Translates user-friendly wizard YAML to FSM configuration at load time
  - WizardHooks - Lifecycle hooks for stage events: on_enter, on_exit, on_complete, on_restart, on_error
  - Navigation Commands - Built-in support for "back"/"go back", "skip", and "restart" navigation
  - Stage Features - Per-stage prompts, JSON Schema validation, suggestions, help text, can_skip, can_go_back, and stage-scoped tools
  - Response Metadata - Wizard progress tracking, current stage info, and available actions
  - Two-Phase Validation - Extraction confidence check followed by JSON Schema validation with graceful degradation
  - State Persistence - Wizard state stored in ConversationManager.metadata for cross-turn persistence

#### Changed
- integration and factory update

#### Fixed
- fixed self-deprecation warnings in tests

### dataknobs-llm [0.3.0]

#### Added
- SchemaExtractor - LLM-based structured data extraction from natural language using JSON Schema
- ExtractionConfig - Configuration for extraction provider, model, and confidence threshold
- ExtractionResult - Result object with extracted data, confidence score, and validation errors
- Multi-Provider Support - Extraction works with Ollama (dev), Anthropic, and OpenAI providers
- Per-Stage Model Override - Stages can specify different extraction models for varying complexity

### dataknobs-common [1.2.0]

#### Added
- EventBus abstraction with Memory/Postgres/Redis backends

### dataknobs-structures [1.0.5]

#### Fixed
- Updated pyparsing API calls to use non-deprecated names (nested_expr, parse_string)

## Release - 2026-01-05

To all packages except legacy, added py.typed markers to enable PEP 561 type checking support for downstream consumers.
Patched versions:
- dataknobs-common [1.1.3]
- dataknobs-config [0.3.3]
- dataknobs-structures [1.0.4]
- dataknobs-utils [1.2.2]
- dataknobs-xization [1.2.3]
- dataknobs-data [0.4.5]
- dataknobs-fsm [0.1.5]
- dataknobs-llm [0.2.4]
- dataknobs-bots [0.2.6]

## Release - 2025-12-26

### dataknobs-xization [1.2.2]

#### Added
- JSON chunking
- Knowledge base ingestion

### dataknobs-data [0.4.4]

#### Added
- Hybrid search types
- Backend hybrid search integration

### dataknobs-bots [0.2.5]

#### Added
- RAGKnowledgeBase hybrid search enhancements


## Release - 2025-12-16

### dataknobs-config [0.3.2]

#### Added
- multi-layered environment-aware configuration support

### dataknobs-bots [0.2.4]

#### Added
- multi-layered environment-aware configuration support
- BotRegistry enhancements
- Deprecated BotManager -- use BotRegistry instead


## Release - 2025-12-15

### dataknobs-data [0.4.3]

#### Added
- Added pgvector backend

### dataknobs-llm [0.2.3]

#### Added
- Implemented per-request LLM overrides

### dataknobs-bots [0.2.3]

#### Added
- Implemented per-request LLM overrides
- Added copy() method to BotContext

## Release - 2025-12-13

### dataknobs-common [1.1.2]

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.

### dataknobs-config [0.3.1]

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.

### dataknobs-structures [1.0.3]

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.

### dataknobs-utils [1.2.1]

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.

### dataknobs-xization [1.2.1]

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.

### dataknobs-data [0.4.2]

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.

### dataknobs-fsm [0.1.4]

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.

### dataknobs-legacy [0.1.1]

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.

### dataknobs-llm [0.2.2]

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.

### dataknobs-bots [0.2.2]

#### Added
- Connected streaming responses; Added middleware access to full response.

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.


## Release - 2025-12-08

### dataknobs-common [1.1.1]

#### Added
- Added testing utilities

### dataknobs-data [0.4.1]

#### Fixed
- Validation constraint fixes

### dataknobs-llm [0.2.1]

#### Changed
- Prompt definition and management enhancements

### dataknobs-bots [0.2.1]

#### Changed
- Leverage the LLM package prompt enhancements
- Added progress tracking and logging middleware
- Added a Multi-Tenant Bot Manager
- Added API exception and dependency management
- Added examples, documentation, and tests

## Release - 2025-11-05

### dataknobs-bots [0.1.0]

#### Added
- created new bots package

### dataknobs-llm [0.1.1]

#### Fixed
- fixed option types and logging for the OllamaProvider

## Release - 2025-11-04

### dataknobs-llm [0.1.0]

#### Added
- created new llm package

### dataknobs-xization [1.1.0]

#### Added
- added markdown chunking utilities

#### Changed
- updated documentation

#### Fixed
- fixed ruff and mypy validation errors; moved md_cli.py to xization/scripts

### dataknobs-data [0.3.2]

#### Fixed
- fixed get_nested_value bug for metadata fields
- fixed intermittent test failures

### dataknobs-fsm [0.1.2]

#### Changed
- moved llm modules and llm-based examples to the llm package

## Release - 2025-10-08

### dataknobs-data [0.3.1]

#### Changed
- Dependency security updates
- Fixed psql backend construction to accept connection_string
- Fixed sql search results to include record storage_id
- various lint and test fixes 

### dataknobs-fsm [0.1.1]

#### Changed
- Dependency security updates

#### Fixed
- updated documentation


## Release - 2025-09-20

### dataknobs-fsm [0.1.0]
- Initial Release

### dataknobs-data [0.3.0]

#### Added
- Fixed ID management in filters; added 'NOT_LIKE' operator
- Enhanced `upsert` method signature to accept just a Record object
  - All database backends now support `upsert(record)` in addition to `upsert(id, record)`
  - Automatically uses Record's built-in ID management (storage_id > id field > generated UUID)
  - Maintains full backward compatibility with existing code

#### Changed
- Enhanced upsert to take just a record and use its ID

#### Fixed
- Fixed to properly skip tests in the absence of services.
- Fixed to properly address services from within the development docker container.

## Releases - 2025-08-31

### Dataknobs project

#### Added and Fixed
- Global developer tools and project documentation

### dataknobs-common [1.0.1]

#### Fixed
- Auto lint and formatting fixes

### dataknobs-structures [1.0.1]

#### Fixed
- Auto lint and formatting fixes

### dataknobs-xization [1.0.1]

#### Fixed
- Auto lint and formatting fixes

### dataknobs-data [0.2.0]

#### Added
- Added SQLite backend
- Added VectorStore abstraction
  - As an integrated feature in Databases
  - As a stand-alone abstraction

#### Fixed
- All ruff lint and mypy errors

## Releases - 2025-08-18

### Dataknobs project

### Added
- New modular package structure
- `dataknobs-structures` - Core data structures
- `dataknobs-utils` - Utility functions
- `dataknobs-xization` - Text processing
- `dataknobs-common` - Shared components
- Migration guide from legacy package

### Changed
- Migrated from Poetry to uv package manager
- Split monolithic package into focused modules
- Improved test coverage and organization

### Deprecated
- Legacy `dataknobs` package (use modular packages instead)

### dataknobs-data [0.1.0] - Initial Release 🎉

#### Added
- **Multiple Storage Backends**: Memory, File, PostgreSQL, Elasticsearch, and S3 support
- **Async-First Architecture**: Native async/await support with connection pooling
- **Advanced Query System**: Rich operators with boolean logic (AND/OR/NOT)
- **Pandas Integration**: Seamless DataFrame conversion and batch operations
- **Ergonomic Field Access**: Dictionary-style (`record["field"]`) and attribute-style (`record.field`) access
- **Schema Validation**: Built-in validation and migration utilities
- **Streaming Operations**: Efficient read/write streaming for large datasets
- **Factory Pattern**: Dynamic backend selection via configuration
- **Example Projects**: Complete sensor dashboard demonstration app
- **Connection Pooling**: Automatic pool management for PostgreSQL and Elasticsearch

### dataknobs-config [0.2.0]

#### Added
- **Factory Registration System**: Register and manage factories at runtime
  - `register_factory()` - Register custom factory instances
  - `unregister_factory()` - Remove registered factories  
  - `get_registered_factories()` - List all registered factories
- **Cleaner Configurations**: Reference factories by name instead of module paths
- **Runtime Substitution**: Swap factories at runtime (useful for testing)

### dataknobs-utils [1.1.0]

#### Added
- **PostgreSQL Enhancements**:
  - `port` parameter for `PostgresDB` class
  - Parameterized query support in `execute()` method
- **Improved Security**: SQL injection protection via parameter binding

### dataknobs-legacy [0.0.16]

#### Changed
- Updated imports to use new modular package structure
- Improved compatibility layer for smooth migration

### Developer Experience Improvements

#### Added
- **`dk` Developer Tool**: Unified command-line interface for development
  - `dk test` - Run tests with automatic service orchestration
  - `dk quality-checks` - Run comprehensive quality checks
  - `dk docs` - Build and serve documentation
  - `dk build` - Build distribution packages
- **Enhanced Testing Infrastructure**:
  - Automatic Docker service management for integration tests
  - Parallel test execution support
  - Improved coverage reporting
  - Test debugging utilities
- **Documentation Improvements**:
  - Comprehensive package documentation
  - Real-world example projects
  - Migration guides

## Legacy Package [0.0.15] - Pre-2025

### Added
- Initial tools, features, and functionality

---

For more details on each release, see the [GitHub Releases](https://github.com/KBS-Labs/dataknobs/releases) page.

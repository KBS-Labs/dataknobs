# Changelog

All notable changes to the dataknobs-bots package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- **Knowledge backends no longer block the event loop on storage I/O.**
  The `S3KnowledgeBackend` performs all S3 operations through an async
  (`aioboto3`) client instead of a synchronous `boto3` client, so reads,
  writes, listings, streaming, and change detection run without stalling
  the running event loop. The `dataknobs-bots[s3]` extra now installs
  `aioboto3` alongside `boto3`. The `FileKnowledgeBackend` offloads its
  filesystem reads/writes to a worker thread. Passing a file-like
  `content` argument to any backend's `put_file` no longer blocks the
  event loop on the read. `RAGKnowledgeBase` document ingest offloads its
  file reads, including the `knowledge_base.{yaml,yml,json}` config probe
  and parse that `load_from_directory` performs when no explicit config is
  supplied. The public method signatures and behavior are unchanged — only
  the threading/transport underneath. (One-time botocore data loading on
  the first S3 client creation per session is an `aioboto3`/`botocore`
  characteristic in the shared session factory and is unchanged.)

- The knowledge ingestion manager now publishes lifecycle events on the
  `ingest:domain:start` / `ingest:domain:end` topics (was a single
  `knowledge:ingestion` completion event). The end event fires on both
  success (`status="completed"`) and failure (`status="failed"`), and a
  start event fires at the head of every run. The former
  `knowledge:ingestion` topic is gone; consumers re-subscribe to one or
  both new topics (a single `EventBus` `pattern="ingest:domain:*"`
  subscription catches both with one handler). The manager's `__init__`
  signature and the `event_bus` cross-replica semantic are unchanged.

- **`RAGKnowledgeBase.close()` only closes collaborators it owns.** A
  vector store and embedding provider built from config are owned and
  closed on `close()` as before. A store or provider injected via
  `RAGKnowledgeBase.from_components(vector_store=…, embedding_provider=…)`
  is caller-owned and left open, so a consumer sharing one store/provider
  across several knowledge bases can close each base independently without
  tearing down a resource the others still depend on. Consumers that build
  the knowledge base from config see no change.

- **`close()` across the bot stack now tears down only collaborators the
  holder owns.** A holder that builds a collaborator from config owns its
  lifecycle and closes it; a holder handed a pre-built collaborator leaves
  it open for its real owner. This makes sharing one backing resource
  across several holders safe — closing one holder no longer tears down a
  collaborator the others still depend on. Applies to:

  - **`DynaBot`** — the knowledge base, memory, reasoning strategy, and
    conversation storage are closed only when built from the bot's config.
    Collaborators injected via `DynaBot.from_components(...)` (or the
    pre-built `DynaBot(llm=...)` constructor) are caller-owned and left
    open, so several bots can share one knowledge base or storage backend
    and each can be closed independently. (The main LLM keeps its existing
    behavior: a provider passed to `from_config(config, llm=...)` is
    caller-owned; one the bot builds itself is closed.)
  - **`MemoryBank`** — a caller-supplied `db` is left open by default;
    `MemoryBank.from_dict(db=None)` builds a db the bank owns and closes.
    A new `owns_db` parameter overrides the inference when a db is built
    elsewhere for a bank's exclusive use.
  - **`VectorKnowledgeSource`** — never closes the knowledge base it
    wraps (the KB is supplied by the caller and typically shared with the
    owning bot and other sources). A new `owns_kb` parameter opts into
    closing a dedicated KB.
  - **`GroundedReasoning`** — closes only the extractor and sources it
    built from config. An extractor or source injected by the caller, and
    the query provider (always the bot's LLM or an injected override), are
    left open. `add_source` gained an `owns` keyword to add a shared
    source without transferring ownership.

  `CompositeMemory` and `HybridReasoning` own the children they compose on
  every construction path (the children are dedicated to the parent, not
  shared), and each child independently protects any backing resource it
  was handed — so they continue to close their children unconditionally.
  Consumers that build everything from config see no behavior change.

  **Migration — default `close()` semantics flipped for direct
  construction.** The new default for a directly-constructed holder is
  *leave the injected collaborator open*, the inverse of the prior
  *close-it* default. If you relied on the old behavior — building a
  resource specifically for one holder and using the holder's `close()`
  to release it — you must now opt back into ownership explicitly, or the
  resource stays open (a leak):
  - `MemoryBank(db=db).close()` no longer closes `db`. Pass
    `MemoryBank(db=db, owns_db=True)` (or `from_dict(..., owns_db=True)`)
    for a db built for that bank's exclusive use. `from_dict(db=None)`
    still builds and owns its db. (Passing `owns_db=False` alongside
    `db=None` is contradictory — the bank holds the only reference — so it
    is ignored with a `UserWarning` and the internally-built db is owned.)
  - `VectorKnowledgeSource(kb).close()` no longer closes `kb`. Pass
    `VectorKnowledgeSource(kb, owns_kb=True)` for a KB dedicated to that
    source.
  - `DynaBot(knowledge_base=kb, memory=…, …).close()` no longer closes the
    pre-built collaborators it was handed. Build the bot from config (which
    owns what it builds), or close the shared collaborators yourself.

  Consumers that build everything from config are unaffected — ownership is
  inferred correctly on the config path.

- Re-platformed `LifecycleHooks` (and via composition,
  `WizardHooks`) onto the new `dataknobs_common.callbacks`
  `CallbackRegistry` substrate. The consumer-facing surface
  (`on_turn_start` / `on_turn_end` registration with chaining and
  per-stage scoping; `trigger_turn_start` / `trigger_turn_end`
  triggers; `turn_start_count` / `turn_end_count` properties;
  `clear()` with drain-in-place identity; `from_config` with
  dotted-path callback resolution) is unchanged. New: a `registry`
  property exposes the underlying `CallbackRegistry` as a
  documented escape hatch — consumers can swap orderings
  (`hooks.registry.set_ordering(PriorityOrdering())`), register
  priority-tagged callbacks
  (`hooks.registry.register("turn_start", cb, priority=-100)`), or
  fan turn-lifecycle events out to an `EventBus`
  (`hooks.registry.also_publish_to(bus, topic_prefix="wizard:")`)
  without monkey-patching. A new `LifecycleHooks.load_config(config)`
  instance method registers callbacks against the existing registry
  in place — used internally by `WizardHooks.from_config` so the
  embedded `LifecycleHooks` constructed in `WizardHooks.__init__`
  survives the config load; consumers caching a
  `hooks.lifecycle.registry` reference (e.g. to install a custom
  ordering or fan-out target) keep that reference valid across
  `WizardHooks.from_config`. `LifecycleHooks` declares
  `Capability.CALLBACK_REGISTRY` via `CapabilityMixin` for
  feature-probe-before-use composition. The `WizardHooks.clear()`
  invariant ("lifecycle instance identity is preserved") now also
  extends to the underlying registry — `hooks.registry is
  hooks.registry` survives `clear()` so consumer customizations
  (custom ordering, fan-out targets) persist across resets.
- Replaced the `aioresponses`-driven HTTP fixture in
  `tests/test_registry_http_backend.py` with an in-process
  `aiohttp.web` test server (`_MockHttpServer`). Every request now
  flows through the real aiohttp client/server stacks — wire-format
  bugs (encoding, multi-valued query params, headers) surface in the
  test instead of in production. Eliminates the temporary
  `aiohttp<3.14` cap on dev deps and unblocks the
  `aiohttp>=3.14.1` floor bumps in `dataknobs-llm` and
  `dataknobs-fsm`. Test-side changes: the auth-header test now
  asserts the actual `Authorization: Bearer ...` header reaches the
  server (previously the assertion was implicit), and the
  no-wire-protocol peek test now asserts an empty server-side query
  mapping rather than scanning the URL string.
- `LifecycleHooks` / `WizardHooks` turn-lifecycle triggers
  (`trigger_turn_start` / `trigger_turn_end`) take a single opaque
  `event: dict[str, Any]` argument; `TurnHookCallback` is
  `Callable[[dict[str, Any]], None | Awaitable[None]]`. The
  wizard publishes canonical event keys `stage`, `phase`,
  `reason`, `manager`, and `state`; adopters attach
  subsystem-specific keys (e.g. the stream-abandonment path
  attaches `state_saved=False`) without extending the protocol
  signature. Hook callbacks read named keys off the event
  payload. Documented in USER_GUIDE.md "Turn-Lifecycle Hooks".
- `WizardReasoning` fires `on_turn_end` on every turn exit, with a
  per-site `reason` discriminator on the event payload:
  `"normal"` from the canonical `finalize_turn` /
  `stream_finalize_turn` save→fire path (including subflow-push
  variants and the streaming counterpart); `"amendment"` /
  `"navigation"` from the `begin_turn` early-return paths;
  `"clarification"` / `"collection_help"` / `"collection_loop"` /
  `"confirmation"` / `"validation_error"` from the
  `process_input` early-return paths; `"abandoned"` (with
  `state_saved=False`) from the stream-abandonment path when the
  consumer calls `aclose()` on the async iterator; and
  `"advance"` (with `manager=None`) from the non-conversational
  `advance()` API. A consumer observing `chat()` and
  `stream_chat()` sees the same fire-points for the same
  conversation outcomes. State-mirroring consumers ignore
  non-advancing turns by filtering on
  `event.get("reason") == "normal"`; observability / audit /
  metric consumers typically observe every exit and tag records
  with the reason. The shared `_fire_turn_end_hook` helper
  resolves the active subflow FSM at every site, so
  `event["stage"]` reflects the deepest pushed subflow at the
  fire-point.
- `make_metadata_inbox_hook` reads `manager` / `state` / `stage`
  from the event payload — matches the
  `Callable[[dict[str, Any]], ...]` trigger contract.

### Added

- `dataknobs_bots.knowledge.events` module — the canonical
  knowledge-layer event topic constants (`INGEST_DOMAIN_START`,
  `INGEST_DOMAIN_END`, `INGEST_METADATA_WRITE`,
  `INGEST_SNAPSHOT_WRITE`, all `Final[str]`), the
  `KnowledgeTriggerPayload` `TypedDict` documenting the ingest-trigger
  payload shape, and the `TenantFilteredCallback` adapter that
  short-circuits an event callback on tenant mismatch. All re-exported
  from `dataknobs_bots.knowledge`.
- `KnowledgeIngestionManager.lifecycle_callbacks` — an in-process
  `CallbackRegistry` that fires `INGEST_DOMAIN_START` at the head of
  every `ingest()` / `ingest_changes()` and `INGEST_DOMAIN_END` at the
  tail (on success **and** failure). When the manager is constructed
  with an `event_bus`, the registry auto-composes
  `also_publish_to(event_bus)` so the lifecycle events fan out to the
  bus for cross-replica observability. Payloads carry `tenant_id` only
  when the manager is tenant-bound. Advertises
  `Capability.INGEST_EVENT_PUBLICATION` / `CALLBACK_REGISTRY` always,
  and `EVENT_BUS_EMISSION` only when constructed with an `event_bus`
  (config-dependent, via `DynamicCapabilityMixin`) — a busless manager
  never fans out, so `require_capability(mgr, EVENT_BUS_EMISSION)`
  reports the truth rather than a false positive.
- `KnowledgeResourceBackend.subscribe_to_changes(bus, *, kinds=None,
  domain_id=None, handler)` and the `changes_subscription(...)` async
  context manager — compose a backend's `key_pattern()` with
  `EventBus.subscribe()` in one call; `kinds` defaults to
  `{KnowledgeKeyKind.CONTENT}` (observe consumer writes, skip
  DK-managed state writes). Default implementations ship on
  `KnowledgeResourceBackendMixin`.
- Backend state-write observability — every backend fires
  `INGEST_METADATA_WRITE` / `INGEST_SNAPSHOT_WRITE` on its
  `state_write_callbacks` `CallbackRegistry` after each metadata /
  snapshot write, via the shared `_fire_state_write` helper on
  `KnowledgeResourceBackendMixin`. Zero-overhead when no callbacks are
  registered. Backends advertise
  `Capability.BACKEND_STATE_OBSERVABILITY` /
  `KEY_PATTERN_FILTERING` / `CHANGE_SUBSCRIPTION` / `CALLBACK_REGISTRY`.
- `IngestOrchestrator` honors an optional `payload["key"]` on trigger
  events: when present it classifies the key via the resolved
  backend's `classify_key(...)` and skips non-`CONTENT` keys (so the
  DK-managed state writes the ingest performs do not re-trigger
  ingestion). Absent `key` proceeds unchanged.
- `BackendKeyDiscriminator(backend)` adapter (frozen dataclass) —
  wraps any `KnowledgeResourceBackend`'s `classify_key` method
  through the generic `Discriminator[str, KnowledgeKeyKind]`
  Protocol from `dataknobs_common`. Use when composing
  backend-key classification with other discriminators
  (payload-field routing, multi-field event-handler dispatch)
  through the generic protocol shape without coupling consumer
  code to the backend interface directly. `frozen=True` gives
  `__eq__` / `__hash__` keyed on the wrapped backend so two
  adapters around the same backend instance compare equal
  (useful for adapter-cache lookups). Exported from
  `dataknobs_bots.knowledge` and
  `dataknobs_bots.knowledge.storage`.
- `KnowledgeKeyKind` enum (`CONTENT` / `METADATA` / `SNAPSHOT` /
  `UNKNOWN`) names the three classes of keys every in-tree
  `KnowledgeResourceBackend` writes. External event sources
  (S3 → EventBridge / SQS / SNS / Lambda; filesystem inotify;
  GCS Pub/Sub) use the enum to filter to the consumer-controlled
  `CONTENT` subtree and skip the DK-managed `METADATA` /
  `SNAPSHOT` writes the ingestion manager performs during
  ingest, which would otherwise create a positive feedback loop.
  Exported from `dataknobs_bots.knowledge` and
  `dataknobs_bots.knowledge.storage`.
- `KnowledgeResourceBackend.classify_key(key) -> KnowledgeKeyKind`
  and `KnowledgeResourceBackend.key_pattern(kind=CONTENT,
  domain_id=None) -> str` — the helper API for source-level
  filtering (`key_pattern`) and per-event filtering
  (`classify_key`). Each first-party backend ships its own
  `key_pattern` returning the backend-native pattern dialect:
  S3 wildcard syntax for `S3KnowledgeBackend` (suitable for
  EventBridge `wildcard` rules or composed into bucket-notification
  `prefix` + `suffix` pairs), a `pathlib.Path.glob`-shaped pattern
  for `FileKnowledgeBackend`, and `""` for `InMemoryKnowledgeBackend`
  (no event-source filter is meaningful in-process; the empty
  sentinel preserves protocol symmetry). `key_pattern(UNKNOWN)`
  raises `ValueError` on the S3 and file backends (fail closed —
  there is no shape for "unrecognized keys"). `classify_key`
  inherits a canonical implementation from
  `KnowledgeResourceBackendMixin` that any out-of-tree backend
  honoring the documented layout gets for free.
- `KnowledgeResourceBackendMixin.METADATA_FILE` /
  `CONTENT_DIR` / `SNAPSHOTS_DIR` `ClassVar[str]` declarations.
  The canonical layout constants live once at the contract layer
  so every in-tree backend resolves them via MRO; an out-of-tree
  backend mixing in `KnowledgeResourceBackendMixin` inherits
  identical values without redeclaring them.
- New docs page **"Event triggers for knowledge backends"**
  (`packages/bots/docs/knowledge/event-triggers.md`) describes
  the layout diagram, the positive-feedback-loop failure mode
  external triggers must avoid, and the wiring recipes per
  source (S3 → EventBridge, S3 bucket notification, filesystem
  inotify, GCS Pub/Sub, and a generic
  `classify_key`-driven fallback). Cross-linked from the
  knowledge-base ingestion guide and `IngestOrchestrator` page.
- `intent_confirm:` wizard stage primitive — declarative block that
  expands at load time into `mode: conversation` +
  `response_template` + (optional) `clarification_template` +
  `intent_detection` + `schema` + `transitions`. Built atop
  `dataknobs_llm.intent` for classifier resolution. Wizard authors
  declare `proposal_template`, an `intents:` map (each intent
  carries a `target`, optional `keywords:` override, optional
  `extract:` field name, optional per-intent `llm_fallback: true`),
  optional `on_no_match: {target?, clarification_template?}` for
  fallback routing and reprompt copy, block-level `llm_fallback:
  true` (shorthand for promoting the classifier to a keyword→LLM
  composite chain), and block-level `negation_filter: true` (wraps
  the resolved classifier in `NegationFilter`); the
  `IntentConfirmSynthesizer` expands the rest. The synthesized
  `intent_detection` block sets `per_intent_booleans: true`, so the
  matched intent writes `state.data[intent_name] = True` (the
  back-compat `state.data["_intent"]` key is still written too).
  Zero new runtime branches — every step runs through existing
  wizard machinery. `IntentConfirmSynthesizer.validate` rejects
  empty / non-mapping `intents`, intent specs that are not mappings,
  intents missing a `target`, and intent names that collide with
  the reserved `_intent` runtime key — all surfaced as
  `ConfigurationError` with the offending stage and intent named.
  After expansion the synthesizer removes the original
  `intent_confirm:` block from the stage dict so the FSM-metadata
  layer carries only the synthesized primitives — no parallel
  source of truth. Documented in USER_GUIDE.md "Wizard-as-advisor:
  intent confirmation".
- `clarification_template:` stage field — optional template rendered
  on re-render of a conversation-mode stage when no extraction or
  intent matched. First render still uses `response_template`;
  subsequent renders consult `clarification_template` when set.
  Populated automatically by `intent_confirm:`'s `on_no_match`;
  hand-rolled conversation-mode stages can use it too.
- Stage-synthesizer registry — `StageSynthesizer` Protocol plus
  `register_stage_synthesizer` / `unregister_stage_synthesizer` /
  `iter_stage_synthesizers` / `validate_no_conflicting_fields`
  exports from `dataknobs_bots.reasoning` (also re-exported from
  `dataknobs_bots.reasoning.wizard_loader` for single-module
  imports), and a `stage_synthesizer_backends:
  Registry[StageSynthesizer]`. `register_stage_synthesizer` uses
  `allow_overwrite=True`; re-registering the same `field`
  overwrites by design (consumers commonly replace the in-tree
  synthesizer with a customized one). The loader runs registered
  synthesizers in a dedicated phase BEFORE `_validate_config` and
  FSM translation, so downstream validator and FSM build code see
  only the normalized shape. `IntentConfirmSynthesizer` is the
  in-tree reference adopter; it auto-registers at module import.
  Documented in USER_GUIDE.md "Shipping your own wizard stage
  primitive".
- `WizardConfigBuilder` recognizes `intent_confirm:` and
  `clarification_template:` on stage configs. `intent_confirm:` is
  carried as a raw dict because the synthesizer expands it before
  the builder's typed `StageConfig` runs.
- `WizardExtractor.detect_intent` dispatches through
  `dataknobs_llm.intent.intent_classifier_backends`. The
  `intent_detection:` block selects a backend via `classifier:`
  (preferred) and optional `classifier_config:` forwarded to the
  backend's factory. The legacy `method: keyword | llm` /
  `llm_fallback: true` shape is promoted automatically at runtime
  (`method: keyword` → `classifier: keyword`; `method: llm` →
  `classifier: llm`; per-intent or block-level `llm_fallback:
  true` → `classifier: composite` with a keyword→LLM chain), so
  existing wizard YAML continues to work unchanged. New YAML should
  use the `classifier:` shape. `negation_filter: true` wraps the
  resolved classifier in `NegationFilter`. When `per_intent_booleans:
  true` is set on the block, the matched intent also writes
  `state.data[intent_name] = True` alongside the existing
  `_intent` key.
- `dataknobs_bots.reasoning.lifecycle.LifecycleHooks` —
  strategy-agnostic turn-lifecycle hook surface
  (`on_turn_start` / `on_turn_end`, with optional per-stage scope).
  Loadable from config via `LifecycleHooks.from_config({...})` with
  dotted-path callback resolution (sync or async callables). Importable
  standalone — adoptable by any `ReasoningStrategy` implementation, not
  wizard-specific. Documented adoption recipe in USER_GUIDE.md
  "Adopting LifecycleHooks in Your Own Reasoning Strategy".
  Public introspection surface: `turn_start_count` /
  `turn_end_count` properties and a `clear()` method that drains
  every registered callback in place (preserving instance identity).
- Wizard turn-lifecycle hook surface: `on_turn_start` and
  `on_turn_end` on `WizardHooks`, following the existing
  `on_enter` / `on_exit` / `on_complete` registration shape.
  `WizardHooks` composes `LifecycleHooks` internally and forwards
  registration / triggering through the embedded instance while
  preserving its own error-handler fan-out (failing hooks still
  reach `on_error` callbacks before re-raising). `on_turn_start`
  fires from `begin_turn` (AFTER per-turn ephemeral-key clear,
  BEFORE early-return dispatch) and symmetrically from `greet` so
  bot-initiated greetings inherit the surface. `on_turn_end` fires
  on every turn exit with a per-site `reason` discriminator (see
  the Changed entry for the full reason table). The embedded
  instance is exposed read-only via the `WizardHooks.lifecycle`
  property for detachment scenarios. `WizardHooks.clear()` drains
  the embedded lifecycle in place alongside the legacy hook lists;
  `WizardHooks.hook_count` reports `"turn_start"` / `"turn_end"`
  counts alongside the legacy keys.
- `WizardReasoning.add_turn_start_hook(callback, *, stage=None)` /
  `WizardReasoning.add_turn_end_hook(callback, *, stage=None)` —
  public runtime-attach surface for turn-lifecycle callbacks.
  Lazy-creates the embedded `WizardHooks` if none was supplied at
  construction, then delegates to `WizardHooks.on_turn_start` /
  `on_turn_end`. Pairs with the canonical fire-points so
  observability / audit consumers can attach hooks without
  re-wiring construction.
- `WizardReasoningConfig.manager_metadata_inbox_key: str | list[str] | None`
  — typed knob that auto-registers an `on_turn_start` hook bridging
  one or more `manager.metadata` keys into `wizard_state.data` at the
  start of every turn. Consume-on-read (popped, not get'd);
  None-as-eviction with the default merge; empty-dict and
  non-mapping payloads tolerated; `greet` inherits the bridge.
  Cross-turn bridge from per-stage sub-strategy output into the
  wizard's transition-eval scope without widening the
  `{data, has, bank, artifact}` safe-eval scope. `None` (default)
  disables the bridge — zero behaviour change.
- `WizardReasoningConfig.inbox_merge_fn` — optional merge function
  for inbox payloads. Defaults to `dict.update` (shallow merge).
  Consumers supply deep-merge or conflict-resolving mergers as
  needed.
- `dataknobs_bots.reasoning.wizard_inbox` module:
  `make_metadata_inbox_hook` factory (for consumers building
  custom variants of the bridge) and `write_to_inbox` helper
  (for writer-side code publishing payloads for the next turn's
  consumption).
- `WizardReasoning` now forwards every construction collaborator
  threaded through `WizardReasoning.from_config(config, **kwargs)`
  (`knowledge_base`, `prompt_resolver`, `prompt_envelope`, every key in
  `reasoning_components`, and any other consumer-supplied kwarg) to the
  per-stage sub-strategy that `WizardResponder._resolve_stage_strategy`
  builds. The forwarding leverages the
  `StructuredConfigConsumer.components` pass-through surface plus the
  new `forwardable_components()` mixin helper (see `dataknobs-common`
  CHANGELOG). `WizardReasoning` declares
  `INTERNAL_COMPONENTS = frozenset({"wizard_fsm"})` so the outer
  wizard's FSM handle is never forwarded to a sub-strategy; every other
  collaborator flows through opaquely. Closes the structural blocker
  for per-stage `reasoning: pipeline` (or other composing) sub-
  strategies that need construction-time collaborators (e.g. a
  knowledge-base-aware `GroundedRetrieval` step). Strictly additive — a
  wizard constructed without extras forwards an empty dict (no-op
  spread on the registry call); the 174 existing direct-ctor call
  sites opt in via the new `_forwarded_components` keyword-only
  parameter on `WizardReasoning.__init__`. Consumer composing
  strategies adopting the same mixin pattern get the same forwarding
  discipline (see USER_GUIDE.md "Building your own composing
  strategy").
- `RAGKnowledgeBase(tenant_id=...)` (also accepted on
  `RAGKnowledgeBaseConfig`) binds a tenant identity to the KB. When set,
  every write auto-stamps `tenant_id` into chunk metadata AND folds it
  into the chunk-id prefix; every read AND-composes
  `{"tenant_id": tenant_id}` into the vector-store search filter.
  Defaults to `None` (single-tenant byte-identical posture — no chunk-id
  derivation change, no read-filter change). Write-side precedence:
  auto-derived bound tenant wins on collision with caller-supplied
  `extra_metadata={"tenant_id": …}` so identity cannot be silently
  re-tagged; read-side precedence inverts (explicit-filter-wins) so
  admin tooling can legitimately read across tenants by passing an
  explicit `filter_metadata={"tenant_id": …}`. The asymmetry matches
  the differing write/read threat models and is documented on
  `_resolve_read_filter`.
- `KnowledgeIngestionManager(tenant_id=…, keyword-only)` binds a
  tenant identity to the manager. Threaded uniformly into every chunk
  the manager writes (via the new `_compose_extra_metadata` helper —
  auto-derived `domain_id` / bound `tenant_id` / TOMBSTONE-swap
  `_generation` token all win over caller-supplied `extra_metadata`)
  AND into every destination-side write/delete filter (via the new
  `_scope_for_tenant` helper applied to the CLEAR_FIRST clear, per-file
  purges, tombstone scope, rollback scope, and reconcile scope). Two
  managers bound to distinct tenants but pointing at the same shared
  `RAGKnowledgeBase` now produce disjoint chunk-id namespaces, do not
  delete each other's rows on CLEAR_FIRST / per-file purges, and do not
  tombstone/un-tombstone each other's rows during TOMBSTONE swaps.
  `tenant_id=None` (default) is the single-tenant byte-identical
  posture.
- `extra_metadata=` keyword-only parameter on
  `KnowledgeIngestionManager.ingest()` and
  `KnowledgeIngestionManager.ingest_changes()`. Mapping merged into
  every chunk's metadata before identity auto-derivation; auto-derived
  identity tags (`domain_id`, the bound `tenant_id`, the `_generation`
  token) win on collision; non-identity keys (`region`, `cohort`, any
  custom tag) are preserved as-is. Symmetric with the
  `extra_metadata` parameter on `RAGKnowledgeBase.ingest_from_backend`.
- `extra_metadata=` and `tenant_id=` keyword-only parameters on every
  direct `RAGKnowledgeBase` entry point — `load_markdown_text`,
  `load_markdown_document`, `load_json_document`, `load_yaml_document`,
  `load_csv_document`, `load_documents_from_directory`,
  `load_from_directory`, and `ingest_from_backend` — so consumers
  reaching for any of the seven entry points get the same shape rather
  than being limited to a subset. The `tenant_id=` convenience kwarg
  folds into `extra_metadata` as `{"tenant_id": tenant_id}`; the same
  bound-tenant precedence applies (auto-derived wins on write boundary).
- `RAGKnowledgeBase._CHUNK_ID_PREFIX_KEYS` — ordered tuple of metadata
  keys (`tenant_id`, `domain_id`, `_generation` by default) folded into
  the chunk-id prefix in declared order, with `source_stem` always last.
  Subclasses rebind to add fold positions (e.g. `(... , "region", ...)`)
  without forking `_derive_chunk_prefix`. Single-tenant single-domain
  consumers (none of the declared keys present in metadata) see the
  historical `(source_stem, "_")` prefix unchanged; multi-segment
  prefixes use the `\x1f` record separator so snake_case-tag collisions
  are impossible.
- `RAGKnowledgeBase._RESERVED_METADATA_KEYS` — frozen set of identity
  tag names (`tenant_id` / `domain_id` / `_generation`) the KB owns at
  the write boundary. Caller-supplied `extra_metadata` entries with
  these keys are shadowed by the auto-derived value; non-reserved keys
  flow through unchanged. Documented in the multi-tenant USER_GUIDE
  section "Reserved vs. Consumer-Extensible Metadata Keys".
- `RAGKnowledgeBase` and `KnowledgeIngestionManager` advertise
  `Capability.TENANT_SCOPED_CHUNKS` via the `CapabilityContract`
  protocol (a chunk-layer Tenancy-family identifier added in the
  `dataknobs-common` `Capability` enum — see common's CHANGELOG).
  Consumers fail-fast at config-load time via
  `kb.supports(Capability.TENANT_SCOPED_CHUNKS)` or
  `mgr.supports("tenant_scoped_chunks")` rather than discovering
  chunk-id UPSERT collisions at first write. Advertisement is
  structural ("the class HAS the chunk-layer code path"), not
  activation-state — whether a specific instance is currently
  tenant-scoping is the natural binding check
  (`kb._tenant_id is not None`). The backend-state-layer
  (`TENANT_SCOPED_STATE`) and concurrency-layer
  (`TENANT_SCOPED_LOCKS`) are deliberately NOT advertised at the chunk
  layer; activation lives at the `KnowledgeResourceBackend` /
  `DistributedLock` layers respectively.
- `FileKnowledgeBackend`, `InMemoryKnowledgeBackend`, and
  `S3KnowledgeBackend` inherit `CapabilityMixin` (via the shared
  `KnowledgeResourceBackendMixin`) so the capability-contract surface
  is uniformly present. Each declares an empty `SUPPORTED_CAPABILITIES`
  set today — per-backend widening (e.g. `STREAMING_READS` on backends
  that implement `stream_file`, `CHANGE_SUBSCRIPTION` /
  `EVENT_BUS_EMISSION` / `KEY_PATTERN_FILTERING` once subscribe/emit
  surfaces ship, `TENANT_SCOPED_STATE` once `set_ingestion_status` /
  `get_checksum` / `has_changes_since` are tenant-scoped at the
  contract layer) is captured in the roadmap rather than declared
  speculatively. Adopters checking `backend.supports(...)` for any
  specific capability get the honest "not advertised" answer.

### Fixed

- Cross-tenant `chunk_id` UPSERT collision in shared
  `RAGKnowledgeBase` instances under the same `domain_id`: two tenants
  ingesting the same `domain_id` through a shared KB previously
  produced N rows (the second ingest UPSERTed the first's chunks in
  place) instead of 2*N rows with disjoint chunk-id namespaces. The
  chunk-id derivation in `_embed_and_store_chunks` now folds every
  present, truthy key from `_CHUNK_ID_PREFIX_KEYS` into the chunk-id
  prefix — `tenant_id` is the first fold position by default, so a
  bound-tenant `KnowledgeIngestionManager` (or a caller threading
  `extra_metadata={"tenant_id": ...}` through `ingest_from_backend`)
  produces distinct chunk ids per tenant. Single-domain single-tenant
  consumers see no change: with `tenant_id` absent the loop yields the
  historical `[domain_id?, generation?, source_stem]` shape
  byte-for-byte.
- Cross-tenant filter-based deletion in `KnowledgeIngestionManager`
  under a shared destination: tenant B's CLEAR_FIRST re-ingest
  previously wiped tenant A's chunks under the same `domain_id`
  (filter was `{"domain_id": domain_id}` only). All filter-based
  mutations on the destination (CLEAR_FIRST, per-file purges,
  TOMBSTONE swap scope, rollback scope, and reconcile scope) now
  AND-compose the bound `tenant_id` via `_scope_for_tenant`, so a
  tenant-bound manager cannot accidentally delete or un-tombstone
  another tenant's rows. Unbound managers (single-tenant) are
  unaffected.

### Security

- Bumped minimum `starlette` requirement (extra: `server`, and the
  matching dev-dependency floor) from `>=1.0.1` to `>=1.3.1` to
  exclude GHSA-82w8-qh3p-5jfq (CVSS 7.5, `request.form()` silently
  ignores `max_fields` / `max_part_size` on
  `application/x-www-form-urlencoded` payloads, enabling a sub-10MB
  DoS), flagged at the floor resolve by the `dependency-update`
  workflow. The bump also sweeps GHSA-jp82-jpqv-5vv3 (CVSS 3.7,
  `request.url` userinfo poisoning on malformed paths missing a
  leading slash, fixed in 1.3.0), GHSA-wqp7-x3pw-xc5r (CVSS 7.5,
  Windows-only `StaticFiles` NTLM credential leak via UNC paths,
  fixed in 1.1.0), and GHSA-x746-7m8f-x49c (CVSS 5.3, `HTTPEndpoint`
  method dispatch using unvalidated client method names when a route
  is registered without an explicit `methods=` argument, fixed in
  1.1.0). The `fastapi>=0.133.0` floor (its `starlette>=0.40.0` no
  upper-cap constraint) permits the bumped floor unchanged.

## v0.7.3 - 2026-06-08

### Added

- `DynaBot.get_steps_of_type(step_cls)` — typed helper that returns
  every reasoning-strategy pipeline step that is an instance of
  `step_cls` as a `list[step_cls]`. Iterates
  `bot.reasoning_strategy.steps` when the strategy is pipeline-shaped;
  returns `[]` when the bot has no reasoning strategy or when the
  strategy has no `steps` attribute. Intended for post-construction
  injection of runtime collaborators that configuration cannot carry.
- `ReasoningStrategy.restore_from_checkpoint(manager, node_metadata)`
  — public hook called by `DynaBot.undo_last_turn` /
  `rewind_to_turn` so a strategy can reinstate per-state buckets it
  persists into a checkpoint node's metadata. Default no-op.
  `WizardReasoning` overrides to restore wizard FSM state from
  `node_metadata["wizard_fsm_state"]`.
- `ReasoningStrategy.undo_to_checkpoint(node_id)` — public hook
  called by `DynaBot.undo_last_turn` / `rewind_to_turn` so a strategy
  can revert node-keyed state. Default no-op. `WizardReasoning`
  overrides to undo each `MemoryBank` it owns.
- `DynaBot.from_config(config, *, reasoning_components=...)` —
  forwards a consumer-supplied mapping into the reasoning strategy's
  `StructuredConfigConsumer.components` channel at construction time.
  Strategies pick up the keys they read (e.g. `ReActReasoning` reads
  `extra_context`, `artifact_registry`, `review_executor`,
  `context_builder`, `prompt_refresher`); unknown keys are silently
  absorbed. Bot-managed components (`knowledge_base`,
  `prompt_resolver`, `prompt_envelope`) raise `ConfigurationError`
  on collision — use their respective config fields instead.

## v0.7.2 - 2026-06-06

### Added

- **`DynaBotConfig.prompt_envelope` selects the user-prompt and
  synthesis-system-prompt envelope style.** `"markdown"` (default)
  renders the auto-context user prompt as `## Knowledge base` /
  `## Conversation history` / `## Question` sections separated by
  `\n\n---\n\n`, and renders the grounded-reasoning synthesis system
  prompt's knowledge-base block as `## Knowledge base\n\n...`. `"xml"`
  reproduces the previous shape byte-for-byte (`<knowledge_base>` /
  `<conversation_history>` / `<question>` blocks separated by `\n\n`,
  and the legacy `<knowledge_base>...</knowledge_base>` synthesis-prompt
  block). `"prose"` renders bare `Label:\n\nbody` sections.
- New `dataknobs_bots.prompts.PromptEnvelope` and
  `PromptEnvelopeStyle` re-exports — a small typed helper used at
  every site that wraps a labeled context block, so the wrap style is
  chosen in one place and matches across the user prompt and the
  synthesis system prompt.
- `KnowledgeBase.format_context`, `RAGKnowledgeBase.format_context`,
  and `ContextFormatter.wrap_for_prompt` accept a keyword-only
  `envelope=` argument. When supplied, the wrapper renders in the
  envelope's style; when omitted, `wrap_in_tags=True` still produces
  the legacy `<knowledge_base>...</knowledge_base>` shape byte-for-byte
  so direct callers are unchanged.

### Changed

- The bot-assembled user prompt and the grounded-reasoning synthesis
  system prompt now default to markdown envelopes. Small
  instruction-tuned models can complete an XML-wrapped input shape by
  emitting a matching wrapper element around their reply (for example
  `<response>...</response>`); switching the default away from XML
  removes that mirroring cue. Model output bytes will shift on the
  next turn for consumers on the default. Pin
  `prompt_envelope: "xml"` to defer the change.
- The `grounded.synthesis.kb_wrapper` library prompt key is no longer
  registered. `GroundedReasoning.build_synthesis_system_prompt` now
  wraps the knowledge-base block through the bot-wide
  `PromptEnvelope`, so the wrap shape is selected by
  `DynaBotConfig.prompt_envelope` instead of by a separate library key.
  Consumers that overrode `grounded.synthesis.kb_wrapper` in a custom
  prompt library should switch to selecting the envelope style.
- **`context_transform` now receives the unwrapped KB body.** A
  consequence of moving the wrap decision into `PromptEnvelope`: the
  bot now asks the knowledge-base layer for an unwrapped body and
  hands that to `context_transform` *before* the envelope wraps it
  (in any style). Pre-fix, the transform saw
  `"<knowledge_base>\n...\n</knowledge_base>"` because the bot wrapped
  before transforming. Consumers whose `context_transform` callable
  pattern-matched on the XML wrappers (e.g. fenced or stripped them)
  must update their transform to operate on the bare body. Memory
  context (`conversation_history`) is unaffected — pre-fix already
  applied the transform to the unwrapped body before wrapping.
- **`prompt_envelope` validation is case-insensitive.** YAML configs
  written by humans now accept `"XML"`, `"Markdown"`, `"PROSE"`, etc.
  Values are normalized to lowercase on the frozen snapshot, so
  downstream lookups continue to match the lowercase enum values.
- **`DynaBot.HybridReasoning` now forwards `prompt_envelope` to its
  grounded child.** A hybrid-strategy bot configured with
  `prompt_envelope: "xml"` (or `"prose"`) had been silently rendering
  the synthesis-prompt KB block with the grounded child's default
  markdown envelope because hybrid did not propagate the collaborator.
  The envelope now reaches the grounded child unchanged.

### Fixed

- The pre-built `DynaBot(llm=provider, prompt_builder=..., ...)`
  constructor now accepts a `prompt_envelope` keyword. Programmatic
  construction (tests, `BotTestHarness`, advanced callers) can pin a
  non-default envelope without going through a config mapping; absent
  the keyword, the `DynaBotConfig` default `"markdown"` applies, so
  every existing call site is unchanged.
- **`ContextPersister.persist()` now correctly persists conversation
  context across the pre-/post-state boundary.** The previous
  implementation read `manager.metadata` (a read-only `@property`)
  and then assigned the mutated dict back to `manager.metadata` —
  which raised `AttributeError: property 'metadata' has no setter`
  on every call against a real `ConversationManager`. The call is
  now routed through `ConversationManager.update_seed_metadata`, so
  the context section is written to the live `state.metadata` (when
  state has been materialized) and to the initial-metadata seed
  bucket (always), with the same replace-not-merge semantic the
  original implementation intended. Behavioural tests against a real
  `ConversationManager` pin both the pre-state and post-state paths.
- **`DynaBot._execute_tools` now routes through
  `ToolRegistry.execute_tool` so the registry's execution tracker is
  populated on real bot turns.** Pre-fix, DynaBot called
  `tool.execute()` directly, bypassing the registry's recording code
  path. Consumers reading `tool_registry.get_execution_history()`
  always saw an empty list on a real turn — most notably
  `ContextBuilder._extract_tool_history`, which surfaces tool history
  into the prompt-rendered context section. The end-to-end chain
  (DynaBot turn → tool execution → tracker → context history) was
  broken at the first step. Dispatch now goes through
  `registry.execute_tool`, whose forwarding semantic was fixed in
  `dataknobs-llm` (`ContextAwareTool` receives `_context` per its
  docstring; plain tools are unaffected). DynaBot's per-tool timing,
  error handling, and `ToolExecution` records on `TurnState` are
  unchanged — the only behavioural shift is that a registry
  constructed with `track_executions=True` now sees a record per
  tool call during a real bot turn.

## v0.7.1 - 2026-06-02

### Added

- **`history_redactions` on every memory backend config — read-time
  citation redaction.** `BufferMemoryConfig`, `SummaryMemoryConfig`, and
  `VectorMemoryConfig` each carry a new
  `history_redactions: list[HistoryRedaction]` field (default empty —
  passthrough); each backend's `get_context()` rewrites assistant-role
  messages on the way out to the prompt-feed. `HistoryRedaction`
  (re-exported from `dataknobs_bots.memory`; canonical home in
  `dataknobs-llm`) is a `(pattern, replacement)` regex spec, applied in
  declared order — list the more specific pattern (a bracketed citation
  header) before the more general bare token. Stored state is never
  mutated: `BufferMemory.messages`, the `SummaryMemory` recent deque,
  and the vector-store rows keep the original text, so direct reads of
  the buffer, exports, and any UI that bypasses `get_context()` see
  un-redacted content. Backend-specific behavior:
  - `SummaryMemory` also applies the same redactions to overflow
    messages before they are summarized, so a citation token in an
    aged-out turn cannot survive in the system-role summary header.
  - `VectorMemory` applies redactions to search-result rows after the
    similarity search, so stored vectors and scoring are unaffected.
    `item["content"]` is the redacted view; `item["metadata"]` aliases
    the live stored row, so `item["metadata"]["content"]` still reads
    the un-redacted text — treat `metadata` as a read-only reference to
    the stored row.
  - `CompositeMemory` inherits the guarantee via delegation: each child
    configured with `history_redactions` redacts on its own path.
    `CompositeMemoryConfig` deliberately does not carry the field.
    Children configured with mismatched policies can land the same
    source message in two different `(role, content)` dedup buckets, so
    configure consistently across children that may surface the same
    content.
- **`DynaBotConfig.conversation_middleware`.** New optional list of
  `ConversationMiddleware` specs (same `{class, params, optional}` shape
  as `middleware`) forwarded to every `ConversationManager` the bot
  constructs. Distinct from `middleware` (bot-turn lifecycle hooks):
  `conversation_middleware` wraps the LLM-call boundary
  (`process_request` / `process_response`), so it can transform the
  request and response that hit the provider. `DynaBot.from_config(...)`
  accepts a symmetric `conversation_middleware=` kwarg that replaces the
  config-driven list with pre-built instances (matching the existing
  `middleware=` kwarg). Pairs with `HistoryRedactionMiddleware` from
  `dataknobs-llm` for deployments where the bot's memory is not the
  redaction surface.

### Changed

- **Middleware and tool specs are validated against their target
  interface at config-load.** A `middleware` spec whose resolved class
  does not subclass `Middleware`, a `conversation_middleware` spec whose
  resolved class does not subclass `ConversationMiddleware`, or a
  `tools` spec whose resolved class does not subclass `Tool` raises
  `ConfigurationError` before the spec's constructor is invoked, so a
  misplaced spec cannot trigger constructor side effects. `optional:
  true` continues to silence transient resolution failures (missing
  module / class, malformed params) but no longer silences a
  class-shape mismatch — a wrong-shape spec is a config-layout error,
  not a transient environment failure, and always raises. The tool
  resolver's wrong-shape error message changes accordingly (from
  `"Resolved class … is not a Tool instance"` to `"Resolved class …
  must subclass …Tool"`).

## v0.7.0 - 2026-05-26

### Added

- **Pluggable backend registries for memory, knowledge bases, and grounded
  sources.** Each subsystem's `create_*_from_config` factory now dispatches
  through a `PluginRegistry`, so 3rd parties can register custom backends
  without modifying core code:
  `register_memory_backend(name, factory)` (discriminator `type`, built-ins
  `buffer`/`vector`/`summary`/`composite`),
  `register_knowledge_base_backend(name, factory)` (discriminator `type`,
  built-in `rag`), and
  `register_source_backend(name, factory)`
  (matched against `GroundedSourceConfig.source_type`, built-ins
  `vector_kb`/`database`). Companion `list_*_backends()`,
  `is_*_backend_registered()`, and `get_*_backend_factory()` helpers are
  exported alongside each. All three `register_*_backend` functions are
  re-exported from the top-level `dataknobs_bots` namespace, mirroring
  `register_strategy`.
- **Typed subsystem sub-configs** — `BufferMemoryConfig`,
  `SummaryMemoryConfig`, `CompositeMemoryConfig`, and `VectorMemoryConfig`
  (exported from `dataknobs_bots.memory`) plus `RAGKnowledgeBaseConfig`
  (exported from `dataknobs_bots.knowledge`), all `StructuredConfig`
  subclasses. Each concrete subsystem class consumes the matching typed
  config (see the corresponding *Changed* entry).
- **`from_components(...)` on the subsystem classes** — `VectorMemory`,
  `SummaryMemory`, `CompositeMemory`, and `RAGKnowledgeBase` expose
  `from_components(config=None, **collaborators)` for assembling an
  instance from already-built collaborators (a pre-built vector store and
  embedder, an LLM provider, child memory strategies) instead of from
  config. The collaborator-adopting path does not own the resources it is
  handed (`close()` leaves caller-owned resources open).
- **Typed `DynaBotConfig`** (`StructuredConfig` subclass,
  `dataknobs_bots.bot.config`). A `DynaBot` now carries a typed
  `bot.config: DynaBotConfig` snapshot — a thin top-level envelope of typed
  scalars plus the documented config sections. The polymorphic subsystem
  sections (`memory`, `knowledge_base`, `reasoning`) and the provider
  section (`llm`) stay raw mappings, dispatched by their discriminator in
  the subsystem registries. `DynaBot.from_config()` accepts either a config
  mapping or a `DynaBotConfig`.
- **`DynaBot.from_components(...)`**, the named alias of the pre-built
  collaborator constructor (`DynaBot(llm=provider, prompt_builder=...,
  conversation_storage=...)`), for assembling a bot from already-built
  collaborators.
- **Embedder credentials are redacted from config `repr`.** Via the
  `StructuredConfig._SENSITIVE_FIELDS` mechanism in `dataknobs-common`,
  `VectorMemoryConfig.api_key` and `RAGKnowledgeBaseConfig.api_key` are
  masked as `'***'` in `repr(config)` (and therefore in logs, tracebacks,
  and pytest failure output). `to_dict()` is never redacted, so
  round-trip construction is unaffected. A credential nested inside a raw
  mapping section — the `embedding` dict's `api_key`, the `vector_store`
  dict's `connection_string`, an `llm` dict's `api_key` (including in
  `SummaryMemoryConfig`, which declares no `_SENSITIVE_FIELDS`) — is also
  masked: common's repr now descends into raw `Mapping`/`list` fields and
  masks interior keys in its default sensitive-key set ∪ the class's
  `_SENSITIVE_FIELDS`.

### Changed

- **`GroundedSourceConfig` is now a frozen `StructuredConfig`.** Its
  `from_dict` is derived from the dataclass fields with a `_normalize_dict`
  hook that preserves the legacy flat declaration shape (the `type` key
  aliases `source_type`; keys outside the reserved set collect into
  `options`). Public behaviour is unchanged; it gains `to_dict` and
  symmetric round-tripping and is immutable (construct a modified copy with
  `dataclasses.replace(...)`).
- **The memory, knowledge-base, and grounded-source factories dispatch via
  their backend registries** instead of inline type branching. Public
  signatures (`create_memory_from_config`,
  `create_knowledge_base_from_config`, `create_source_from_config`) and the
  `ValueError` raised on an unknown type are unchanged.
- **The concrete memory and knowledge subsystem classes are now
  `StructuredConfigConsumer`s.** `BufferMemory`, `VectorMemory`,
  `SummaryMemory`, `CompositeMemory`, and `RAGKnowledgeBase` carry a typed
  `self.config` and build through the shared construction lifecycle, and
  the memory registry registers the classes directly (the transitional
  per-backend builder functions are gone). Config-driven construction is
  unchanged: `await create_memory_from_config({...})`,
  `await create_knowledge_base_from_config({...})`,
  `await VectorMemory.from_config({...})`, and
  `await RAGKnowledgeBase.from_config({...})` keep their exact signatures
  and behavior (the async warmup classes expose `from_config` as a
  lifecycle-faithful async delegator that runs `_ainit`), and
  `BufferMemory(max_messages=...)` still works. Direct construction from
  pre-built collaborators moves from positional/keyword constructors to
  `from_components(...)` (see the *Added* entry); e.g.
  `SummaryMemory(llm_provider=p, recent_window=2)` becomes
  `SummaryMemory.from_components({"recent_window": 2}, llm_provider=p)`,
  `CompositeMemory([m1, m2])` becomes
  `CompositeMemory.from_components(strategies=[m1, m2])`, and
  `RAGKnowledgeBase(vector_store=vs, embedding_provider=ep,
  chunking_config={...})` becomes
  `RAGKnowledgeBase.from_components({"chunking": {...}}, vector_store=vs,
  embedding_provider=ep)` (the `chunking_config` / `merger_config` /
  `formatter_config` constructor arguments are now the `chunking` /
  `merger` / `formatter` config keys, with pre-built `chunker` /
  `merger_config` / `formatter_config` accepted as `from_components`
  collaborators). The
  `owns_llm_provider` constructor flag is gone: ownership now follows the
  construction path (a dedicated `llm` config section is owned; an injected
  provider is not). Typing/contract refactor only — no runtime, retrieval,
  or ownership-semantics behavior changes.
- **`DynaBot` is now a `StructuredConfigConsumer` and builds through the
  shared async construction lifecycle** (`from_config` → `from_config_async`
  → `__init__` → `_setup` → `_ainit`). `DynaBot.from_config(config, *,
  llm=None, middleware=None)` keeps its exact signature and behavior (now a
  parity-guarded async delegator); the direct constructor
  `DynaBot(llm=provider, prompt_builder=..., conversation_storage=..., ...)`
  is preserved verbatim as the pre-built collaborator shape. Construction is
  a typing/contract refactor only — no runtime, reasoning, or dispatch
  behavior changes.
- **`config/` subcomponent dataclasses now subclass `StructuredConfig`.**
  `ToolEntry`, `TemplateVariable`, `ConfigTemplate`, `ConfigVersion`,
  `DraftMetadata`, `ComponentSchema`, and the wizard-builder configs
  (`TransitionConfig`, `IntentDetectionConfig`, `ContextGenerationConfig`,
  `StageConfig`, `WizardConfig`) inherit
  `dataknobs_common.structured_config.StructuredConfig`, so `from_dict` is
  derived from the dataclass fields (recursing into nested sub-configs —
  `StageConfig`'s transitions/intent/context and `WizardConfig`'s stages
  are rebuilt automatically, replacing the hand-walked deserialization).
  `ComponentSchema` and `WizardConfig` gain a `from_dict` they previously
  lacked. Serialized output is unchanged: classes whose `to_dict` omits
  defaults or renames keys (e.g. `DraftMetadata`'s `id`) keep that
  bespoke `to_dict`, and frozenset/tuple fields are restored on load via
  `__post_init__`.
- **`TemplateVariable`, `ConfigTemplate`, `ConfigVersion`, `DraftMetadata`,
  and `ComponentSchema` are now frozen** (the wizard-builder configs
  already were). They are immutable-by-design value objects; construct a
  modified copy with `dataclasses.replace(...)` instead of assigning to
  attributes.

### Security

- Bumped minimum `starlette` requirement (extra: `server`, and the
  matching dev dependency) from `>=0.49.1` to `>=1.0.1` to exclude
  PYSEC-2026-161 / GHSA-86qp-5c8j-p5mr — missing Host-header
  validation that poisons `request.url.path` and can bypass
  path-based authentication. Flagged at the floor resolve by the
  `dependency-update` workflow. Because `1.0.1` is a major release and
  `fastapi <0.133.0` capped `starlette<1.0.0`, the coupled `fastapi`
  floor was bumped from `>=0.120.1` to `>=0.133.0` (the lowest fastapi
  whose starlette constraint permits 1.x) in both the `[server]` extra
  and the dev group. `registry.server` uses only FastAPI's own API
  surface and never imports `starlette` directly, so the major bump is
  insulated. The new floor preserves the prior sweep of
  GHSA-7f5h-v6xp-fcq8 (CVSS 7.5) and GHSA-2c2j-9gv5-cj73 (CVSS 5.3,
  0.49.1).

## v0.6.22 - 2026-05-19

### Added

- **`IngestOrchestrator(manager_resolver=...)` + `IngestionManagerResolver`.**
  An injectable async resolver seam for multi-tenant deployments:
  the orchestrator calls `manager_resolver(tenant_id=..., domain_id=...)`
  once per trigger event (tenant/domain parsed from the payload) and
  dispatches to the returned per-tenant `KnowledgeIngestionManager`
  (its own KB backend prefix / `vector_partition` / embedder).
  `ingestion_manager=` and `manager_resolver=` are mutually exclusive
  and exactly one is required (`ValueError` otherwise); the static
  single-`ingestion_manager` path is unchanged. `IngestionManagerResolver`
  is a `@runtime_checkable` `Protocol` exported from
  `dataknobs_bots.knowledge`. The trigger payload gains an optional
  `tenant_id` (absent ⇒ `None` passed to the resolver); a present
  non-string `tenant_id` fails closed (logged + trigger skipped)
  rather than being routed or coerced, since a misidentified tenant
  is a cross-tenant data leak.

### Changed

- **`IngestOrchestrator` per-domain lock key is now tenant-scoped:**
  `f"ingest:{tenant_id or '-'}:{domain_id}"` (was `f"ingest:{domain_id}"`)
  so two tenants sharing a `domain_id` do not false-share one lock
  under a cross-replica backend. Single-tenant triggers (no `tenant_id`)
  degrade to the stable key `ingest:-:<domain_id>`. With the default
  process-local `InProcessLock` this is invisible. Deployments using a
  cross-replica lock (`{"backend": "postgres", ...}`) should note that
  during a rolling upgrade old and new replicas briefly compute
  different keys for the same single-tenant domain, momentarily
  relaxing cross-replica serialization for that domain until all
  replicas are upgraded.

## v0.6.21 - 2026-05-18

### Added

- **`KnowledgeResourceBackend.list_changes_since(domain_id, version)
  -> ChangeSet`** — file-level diff (added / modified / deleted +
  the current canonical version) between the current knowledge base
  and the snapshot identified by `version` (a `get_checksum()`
  value). `has_changes_since` is now its degenerate case
  (`not (await list_changes_since(...)).is_empty`) rather than a
  separately-implemented sibling.
- **`ChangeSet`** (frozen dataclass: `added` / `modified` /
  `deleted` / `version`, with `is_empty`) and
  **`InvalidVersionError`** (raised when a version predates a
  backend's snapshot retention; consumers fall back to a full
  re-ingest) — exported from `dataknobs_bots.knowledge` and
  `dataknobs_bots.knowledge.storage`.
- **`KnowledgeResourceBackendMixin`** — the shared canonical
  change-detection algorithm (`get_checksum` / `has_changes_since`
  / `list_changes_since` over `list_files()` plus a `_load_snapshot`
  seam). All in-tree backends inherit it; out-of-tree backends mix
  it in for correct behaviour for free. All three in-tree backends
  retain per-version snapshots so `list_changes_since` is a minimal
  file-level diff: `InMemoryKnowledgeBackend` (in-process map),
  `FileKnowledgeBackend` (`_snapshots/<version>.json` written after
  every mutation), and `S3KnowledgeBackend` (snapshot objects, or the
  metadata object's own S3 version history — see
  `change_detection_mode` below). An out-of-tree backend that does
  not override `_load_snapshot` still gets correct (full, non-minimal)
  change *detection* via the version-equality short-circuit.
- **`S3KnowledgeBackend(change_detection_mode=...)`** (also via
  `from_config`, default `"snapshot"`) selects how per-version
  snapshots are resolved: `"snapshot"` writes a small
  `{path: checksum}` object under `{domain}/_snapshots/<version>.json`
  after every mutation (self-contained, any bucket); `"s3_versioning"`
  writes no extra objects and instead walks the metadata object's own
  S3 version history (`ListObjectVersions`) — requires bucket
  versioning enabled, and with it disabled a stale version safely
  falls back to a full re-ingest. An unrecognized mode raises
  `ValueError` (fail closed).
- **`IngestOrchestrator` trigger-payload dispatch.** The trigger
  event payload now selects the ingest entry point: `since_version`
  → `ingest_changes` (per-file delta), `force_full` →
  `ingest(swap_mode=CLEAR_FIRST)` (full re-ingest), otherwise the
  unchanged `ingest_if_changed(last_version)` default. `since_version`
  takes precedence over `force_full`. Payloads using only
  `domain_id` / `last_version` are byte-for-byte unchanged.
- **`IngestionStatus.SWAPPING`** — set by the `TOMBSTONE` swap path
  while the new generation is written; a crash here leaves the
  domain in this state with the in-flight token recoverable.
- **Interrupted-swap auto-reconciliation + `KnowledgeIngestionManager.
  reconcile(domain_id) -> bool`.** A process crash between the upsert
  and the commit of a `TOMBSTONE` swap leaves the domain in
  `SWAPPING` with the old generation tombstoned-but-intact and orphan
  new-generation chunks possibly present. The next `ingest()` /
  `ingest_changes()` for that domain now reconciles *before* applying
  anything — restoring the previous generation to visibility and
  dropping exactly the crashed swap's orphans by its persisted
  token — so residue never accumulates and unrelated files are never
  left hidden. `reconcile()` exposes the same recovery as an
  idempotent one-shot for domains that will not be re-ingested soon
  (returns `True` if it reconciled, `False` if there was nothing to
  do). Backed by a new `KnowledgeBaseInfo.generation: str | None`
  field (round-trips through `to_dict`/`from_dict`) and a kw-only
  `generation=` parameter on `KnowledgeResourceBackend.
  set_ingestion_status` (always written through, so any non-SWAPPING
  transition clears a stale token); implemented by the in-memory,
  file, and S3 backends.
- **`KnowledgeIngestionManager.ingest_changes(domain_id,
  since_version, *, progress_callback=None, config=None)`** —
  per-file delta re-ingest. Diffs the source against
  `since_version` (a `get_checksum`/`get_current_version` value),
  purges chunks for deleted *and* modified files, then re-embeds
  only the added/modified files through the same internal apply
  path as a full `ingest()` — so swap semantics cannot diverge
  between the full-domain and per-file routes. An S3 `PutObject`
  on one file in a 100-file corpus now re-embeds one file, not
  the whole corpus. If `since_version` predates the backend's
  snapshot retention (`InvalidVersionError`) it falls back to a
  full re-ingest after a warning — never a silent skip.
- **`IngestionResult.files_deleted`** — count of source files
  whose chunks were removed because the file no longer exists at
  the source (populated by `ingest_changes`; `0` for a full
  `ingest`). Included in `to_dict()` and the `knowledge:ingestion`
  event payload.
- **`RAGKnowledgeBase.ingest_from_backend(file_filter=)`** —
  optional keyword-only `Callable[[KnowledgeFile], bool]`
  predicate, evaluated after the pattern match, restricting
  enumeration to a subset of the backend's files. `None`
  (default) is unchanged behavior. This is the seam
  `ingest_changes` uses to re-embed only the changed files
  through the full pattern/chunking pipeline.
- **`IngestSwapMode`** (`CLEAR_FIRST` / `APPEND` / `TOMBSTONE`)
  plus a keyword-only `swap_mode=` on
  `KnowledgeIngestionManager.ingest()` and `ingest_changes()`
  (exported as `dataknobs_bots.knowledge.IngestSwapMode`).
  `TOMBSTONE` is a crash-safe re-ingest: the existing (scoped)
  chunks are marked `_stale` (hidden from reads), the new
  generation is ingested under distinct generation-keyed chunk
  ids so it never overwrites the old rows, and the old
  generation is physically retired **only on a clean commit** —
  on a raised error or partial-error ingest the rollback drops
  the new generation by its token and restores the old one. The
  old generation is never overwritten or deleted before the new
  one commits, so a crash, a raised error, or a racing
  same-domain re-ingest always leaves a fully restorable
  previous generation (unlike the `CLEAR_FIRST`
  delete-then-insert). A crash mid-swap leaves the domain in
  `IngestionStatus.SWAPPING`, auto-reconciled by the next ingest
  (or `KnowledgeIngestionManager.reconcile`). Honored identically
  by all in-tree vector stores (Memory, FAISS, PgVector, Chroma);
  `ingest_changes(swap_mode=TOMBSTONE)` scopes the swap to
  exactly the changed/deleted files. A transient in-swap read
  window remains (closing it needs a generation pointer-flip,
  a future mode).
- **`RAGKnowledgeBase.query(..., include_stale=False)`** and
  **`hybrid_query(..., include_stale=False)`** — a single shared
  read chokepoint hides chunks tombstoned by an in-progress
  `TOMBSTONE` swap on **both** read paths (vector search and
  hybrid, native and client-side fusion); `include_stale=True`
  returns them. `service.py` / retrieval inherit this through
  `query` / `hybrid_query`.
- **`RAGKnowledgeBase.update_metadata_where(filter, set_)`** —
  delegates to the vector store's filter-keyed bulk metadata
  merge; the destination-side primitive the `TOMBSTONE` swap
  uses to mark (and, on rollback, un-mark) a generation without
  enumerating ids.
- **Optional embedder rate-limit seam.**
  `KnowledgeIngestionManager(__init__, rate_limiter=)` and the
  keyword-only `RAGKnowledgeBase.ingest_from_backend(...,
  rate_limiter=)` accept a
  `dataknobs_common.ratelimit.RateLimiter`. When set, every
  per-chunk embed on the ingest path is preceded by
  `await rate_limiter.acquire("embed")`, so a rate-limited
  embedding provider (e.g. a hosted API) cannot fail a whole
  ingest under burst. The manager threads its `rate_limiter`
  through to the embed core for every swap mode. `None` (the
  default) is byte-for-byte the prior behaviour — no pacing,
  correct for a local Ollama embedder.

### Changed

- **`KnowledgeBaseInfo.version`** is now documented as a
  cache-invalidation / display counter only and is **no longer the
  change-detection key** (it is still incremented on every change).
  Change detection uses the canonical content snapshot
  (`get_checksum`). **`KnowledgeIngestionManager.get_current_version()`**
  consequently returns the canonical snapshot identity (a
  `get_checksum` value), not the monotonic counter — so capturing
  it and passing it back to `ingest_if_changed(last_version=...)`
  is now a correct round-trip.
- **`IngestOrchestrator(__init__)`** accepts a new optional
  `lock: DistributedLock | None = None` parameter **and** a
  configuration-driven `lock_config: dict | None = None`
  alternative. Per-domain serialization of ingest triggers is
  backed by a `dataknobs_common.locks.DistributedLock` (keyed
  `ingest:<domain_id>`) instead of an internal `asyncio.Lock`.
  Supply a pre-built lock via `lock=`, or let the orchestrator
  resolve one through the shared `create_lock` factory by passing
  `lock_config={"backend": "postgres", ...}` — so a multi-replica
  deployment selects a cross-replica backend by configuration
  without writing code (no lock logic lives in `dataknobs-bots`).
  The two are mutually exclusive (passing both raises
  `ValueError`); an unknown `lock_config` backend raises
  `ValueError` (fail closed). The default — neither supplied — is
  `InProcessLock()`, process-local and behaviour-identical to
  prior releases for single-replica deployments. Multi-replica
  deployments must configure a cross-replica lock; a process-local
  lock cannot serialize across replicas. The built-in Postgres
  advisory-lock backend (`lock_config={"backend": "postgres", ...}`)
  provides cross-replica serialization out of the box; other backends
  remain registry-pluggable via
  `dataknobs_common.locks.lock_backends`.
- **`KnowledgeResourceBackend.set_ingestion_status`** accepts
  `IngestionStatus | str` (Protocol + memory / file / S3
  backends). The typed enum is the preferred form; legacy
  string values still work and are normalized internally. An
  unrecognized status string now raises
  `dataknobs_common.exceptions.ValidationError` (was a bare
  `ValueError`) — the message enumerates the accepted values, and
  the type is a `DataknobsError`, **not** a `ValueError` subclass,
  so a bare `except ValueError` no longer silently swallows an
  invalid-status bug. Domain-not-found still raises `ValueError`.
  No in-tree caller catches `ValueError` around status
  normalization, so this is contract-tightening only.
- **`RAGKnowledgeBase.count()` excludes tombstoned chunks by
  default.** A mid-`TOMBSTONE`-swap `count(filter)` previously
  delegated straight to the store and reported old+new (≈double)
  while `query()`/`hybrid_query()` only returned the new
  generation. `count()` now returns the read-visible count
  (`count(filter) − count(filter ∧ _stale=True)`, two store-agnostic
  counts); the new kw-only `include_stale=True` restores the prior
  single delegated count (every stored chunk). The numbers differ
  **only** while a swap is in flight; outside a swap there are no
  `_stale` chunks and the result is unchanged.

### Deprecated

- **`KnowledgeIngestionManager.ingest(clear_existing=)`** — pass
  `swap_mode=` (`IngestSwapMode`) instead. `clear_existing=True`
  maps to `CLEAR_FIRST`, `False` to `APPEND`; passing the
  argument emits a `DeprecationWarning`. With neither argument
  set the default is unchanged (`CLEAR_FIRST`), so existing
  callers that omit it are unaffected.

### Fixed

- **`get_checksum()` → `has_changes_since()` round-trip no longer
  spuriously re-ingests.** `has_changes_since` (and so
  `KnowledgeIngestionManager.ingest_if_changed`) previously compared
  the monotonic `KnowledgeBaseInfo.version` counter while
  `get_checksum()` returned a content-snapshot hash — different
  value spaces, so a consumer pairing the two (the intuitive,
  documented usage) always saw "changed" and re-ingested the entire
  domain on every check. Both now derive from the canonical content
  snapshot, so an unchanged knowledge base correctly reports no
  changes across all in-tree backends (memory / file / S3).
- **`IngestOrchestrator` multi-replica race made honest.** The
  previous `asyncio.Lock`-per-domain provided no protection across
  processes, yet the class docstring implied per-domain
  serialization unconditionally. The docstring now states the
  serialization scope is exactly the scope of the injected lock and
  that multi-replica deployments must inject a cross-replica lock.
- **`IngestOrchestrator` per-domain lock-map leak.** The internal
  `dict[str, asyncio.Lock]` was never evicted, so every distinct
  `domain_id` grew it unbounded for the lifetime of the
  orchestrator. The injected `InProcessLock` reference-count evicts
  its key map, closing the leak.
- **`IngestSwapMode.TOMBSTONE` re-ingest is now genuinely
  crash-safe.** Chunk ids were deterministic, so a re-embedded
  file's new chunks upserted *over* the tombstoned old rows in
  place — clearing their `_stale` mark and destroying the old
  generation the instant the new one was written. TOMBSTONE was a
  no-op for the dominant re-ingest case (any file whose content
  changed), a mid-swap crash or partial-error left freshly written
  chunks live with no `_stale` key (leaked partial generation), and
  an `ingest_changes` rollback un-tombstoned the *whole* swap scope
  — resurrecting files that had been deleted at the source. Each
  swap now mints a `uuid4` generation token folded into the new
  chunks' ids and stamped on their metadata (`_generation`), so the
  two generations coexist physically until a clean commit. Rollback
  (raised failure *or* partial error) drops exactly the new
  generation by its token, restores the modified files' old
  generation to visibility, and unconditionally purges files
  deleted at the source (never resurrected). On a clean commit the
  old generation is physically retired. APPEND / CLEAR_FIRST id
  derivation is byte-for-byte unchanged (the token is opt-in by
  presence), so single-domain consumers and existing populated
  stores are unaffected.
- **Native hybrid fusion no longer under-returns mid-swap.**
  `hybrid_query(fusion_strategy="native")` requested exactly `k`
  from the store's `hybrid_search` and *then* dropped tombstoned
  rows, so when `_stale` chunks ranked in the top `k` it returned
  fewer than `k` visible results during a `TOMBSTONE` swap. Both the
  vector and native-hybrid read paths now share a single
  `_fetch_drop_stale_truncate` helper that over-fetches
  `k * _STALE_OVERFETCH` before the stale gate and truncates to `k`,
  so a swap in progress no longer shrinks native-fusion result
  sets. (`_is_stale`'s `None`-guard was also tightened from a
  truthiness check to an explicit `is not None` — same result for
  every real input, but it correctly documents that the guard
  protects against a metadata-less row, not an empty dict.)

### Security

- Bumped the `[server]` extra's and dev group's minimum `fastapi`
  requirement from `>=0.110.0` to `>=0.120.1` and added an explicit
  `starlette>=0.49.1` floor to exclude starlette versions affected
  by GHSA-7f5h-v6xp-fcq8 (CVSS 7.5) and GHSA-2c2j-9gv5-cj73
  (CVSS 5.3), both fixed in starlette 0.49.1. `starlette` reaches
  the package only transitively via `fastapi`, but `fastapi`'s own
  lower bound on `starlette` never rises to the patched version, so
  an explicit floor is required to guarantee a safe `starlette` for
  all resolvers — not only the floor-resolve audit. The `fastapi`
  bump is required for graph satisfiability: `fastapi 0.110.0`
  capped `starlette<0.37.0`, and `0.120.1` is the lowest `fastapi`
  whose constraint (`starlette<0.50.0,>=0.40.0`) permits 0.49.1.
  Surfaced by the floor-resolve OSV audit in the
  `dependency-update` workflow.

## v0.6.20 - 2026-05-13

### Added

- **`Registration.metadata`** — `dict[str, Any]` field on
  `dataknobs_bots.registry.Registration` for cross-cutting context
  (`tenant_id`, audit info, feature flags) that lands in the storage
  backend's ``metadata`` column rather than mixed into the config
  payload.  Round-trips through `to_dict` / `from_dict` and the HTTP
  wire protocol.

- **`RegistryBackend.register(..., metadata=...)`** — kw-only
  parameter routes caller-supplied metadata to the backend's
  metadata channel.  Implemented by `InMemoryBackend`,
  `DataKnobsRegistryAdapter`, and `HTTPRegistryBackend`.

- **Registry filter / pagination surface** on `RegistryBackend`:
  - `list_all(*, status=None, filter_metadata=None, sort=None,
    limit=None, offset=None)` — list with optional status equality,
    equality filter over the metadata column, sort spec, and
    limit/offset pagination.
  - `list_active(...)` / `list_inactive(...)` — symmetric
    convenience wrappers over `list_all` with the status pinned.
  - `count_all(*, status=None, filter_metadata=None)` — routed
    through `AsyncDatabase.count(query)` so backends with pushdown
    counts (`SELECT COUNT(*) WHERE ...`) benefit transparently.
  - `count(*, filter_metadata=None)` / `count_inactive(...)` —
    pinned-status counterparts.
  - `stream(*, status=None, filter_metadata=None, config=None)` —
    async-iterator surface for large tenant populations, yields
    `Registration` instances one at a time.

- **`BotRegistry` surfaces the new metadata / filter / pagination
  surface** so consumers don't drop to ``registry._backend``:
  - ``register(..., metadata=...)`` threads ``metadata`` to the
    backend's metadata channel.
  - ``list_bots(*, filter_metadata=None, sort=None, limit=None,
    offset=None)`` — no-kwarg form returns active bot IDs as
    before; any kwarg routes through ``list_active`` for pushdown
    filtering.
  - ``list_registrations(*, status=None, filter_metadata=None,
    sort=None, limit=None, offset=None)`` — new method surfacing
    full `Registration` objects (timestamps / status / metadata).
  - ``count(*, filter_metadata=None)`` — tenant-scoped counts.

- **`HTTPRegistryBackend` wire-protocol extensions** — optional
  query parameters on `GET /configs`:
  `?filter_metadata=<URL-encoded JSON object>` (sorted keys for
  deterministic cache lines), `?status=<value>`,
  `?sort=<field>[:asc|desc]` (repeatable; wire order is tie-break
  order), `?limit=<int>`, `?offset=<int>`.  Schema is **additive
  optional**: servers that recognize a parameter honor it; servers
  that don't ignore it and return the broader list.  The client
  defensively re-applies idempotent filters (`filter_metadata`,
  `status`, `sort`) after parsing the response; `limit`/`offset`
  are intentionally NOT re-applied client-side (re-offsetting a
  server-paginated window would drop live rows).

- **`POST /configs/{bot_id}/deactivate`** — new server-side
  endpoint that routes directly to ``RegistryBackend.deactivate``.
  Lets HTTP clients soft-delete without first issuing
  ``GET /configs/{bot_id}`` (which bumps ``last_accessed_at``).
  Returns ``204 No Content`` on success or ``404 Not Found``.

- **`create_registry_router(backend)`** — reference FastAPI router
  in `dataknobs_bots.registry.server` exposing `RegistryBackend` as
  the wire protocol that `HTTPRegistryBackend` speaks.  Consumers
  can stand up a config service backed by any `RegistryBackend`
  (`InMemoryBackend`, `DataKnobsRegistryAdapter` over
  Postgres/SQLite/S3, …) with one line of glue.  FastAPI is an
  optional dependency: importing the module without it installed
  succeeds; calling `create_registry_router` raises `ImportError`
  with an install hint (`pip install 'dataknobs-bots[server]'`).
  Protocol is pinned on both sides by client and server test
  suites — drift breaks both.

- **`ArtifactRegistry.query`** — kw-only `filter_metadata=`,
  `sort=`, `limit=`, `offset=` parameters.  Filter / sort push down
  to the database query so SQL backends can use indexes.  Pagination
  is applied **after** the latest-pointer dedup pass (dual-write
  storage shape — pre-dedup row count diverges from post-dedup
  artifact count, so a pushdown ``LIMIT`` is unsafe).  Existing
  positional parameters (`artifact_type`, `status`, `tags`,
  `filters`) unchanged.

- **`ArtifactRegistry.count`** — new method mirroring `query`
  parameter-for-parameter (minus sort/limit/offset).  Equivalent
  to ``len(await registry.query(...))`` after dedup.

- **`RubricRegistry.list_all` / `RubricRegistry.get_for_target`** —
  kw-only `filter_metadata=`, `sort=`, `limit=`, `offset=`.  Same
  post-dedup pagination policy as `ArtifactRegistry.query` (same
  dual-write storage shape).

- **`RubricRegistry.count_for_target` / `RubricRegistry.count_all`**
  — new count methods mirroring the corresponding list/get methods.

- **`GeneratorRegistry.list_definitions`** — kw-only
  `filter_metadata=`, `sort=`, `limit=`, `offset=`.  Unlike the
  dual-write registries, `GeneratorRegistry` writes a single row
  per generator id — no pointer/snapshot divergence — so
  limit/offset push down to the database directly.

- **`GeneratorRegistry.count_definitions`** — new method that routes
  through `AsyncKeyedRecordStore.count`, letting backends with
  pushdown counts skip row materialization.

### Changed

- **`DataKnobsRegistryAdapter`, `ArtifactRegistry`, `RubricRegistry`,
  and `GeneratorRegistry` now compose `AsyncKeyedRecordStore`** (from
  `dataknobs-data`) instead of building `Record(...)` instances
  inline.  The store's
  ``(T) -> (data, metadata)`` serializer signature makes the
  metadata channel part of the function's type, so a future change
  to a model can't accidentally drop the metadata channel without a
  type-visible diff at the serializer site.  Public surface
  preserved; the `DataKnobsRegistryAdapter` stored shape differs —
  see Migration below.

### Fixed

- **`DataKnobsRegistryAdapter` now persists caller-provided
  metadata to the `Record.metadata` column.**  Previously the
  metadata column was always empty (there was no
  `Registration.metadata` field), rendering `metadata.X` filters
  and the Postgres metadata GIN index unreachable.  Multi-tenant
  consumers can now use `filter_metadata={"tenant_id": ...}` to
  scope `list_active` / `list_all` queries.

- **`ArtifactRegistry` and `RubricRegistry` now persist artifact /
  rubric `metadata` to the `Record.metadata` column** (latent
  defect — no consumer had hit it yet).

- **`GeneratorRegistry` no longer silently routes definition
  fields into the `data` column under a `metadata` variable
  name.**  The pre-fix code passed a local variable named
  ``metadata`` positionally to ``Record(...)``, but ``Record(...)``'s
  first positional is ``data`` — so the schema/version/id fields
  landed in the data column and the record's metadata column was
  never populated.  Migrating to `AsyncKeyedRecordStore` removes
  the inline `Record(...)` call, so the variable-name shadow
  cannot recur and `GeneratorDefinition.metadata` lands in the
  correct column.

- **`DataKnobsRegistryAdapter.count()` no longer materializes
  every active row** to compute its result.  It now routes through
  `_db.count(query)`, so backends with `SELECT COUNT(*)` pushdown
  return without row materialization.

- **`HTTPRegistryBackend.register` and `.deactivate` no longer
  issue touching reads.**  Previously both methods called
  ``await self.get(bot_id)`` first — the corresponding
  ``GET /configs/{bot_id}`` route bumps ``last_accessed_at`` per
  the `get` protocol contract, so every re-register and every
  soft-delete contaminated the user-activity signal that timestamp
  is supposed to carry.  `register` now issues a single
  ``PUT /configs/{bot_id}`` (upsert); `deactivate` calls the new
  ``POST /configs/{bot_id}/deactivate`` endpoint.

- **`ArtifactRegistry.revise` / `set_status` / `submit_for_review`
  are now serialized per artifact id**, closing an in-process
  read-modify-write race.  Two concurrent ``revise(id, …)`` callers
  could both read ``v1.0.0``, both compute ``v1.0.1``, and both
  write the same snapshot key — last-write wins and the losing
  revision silently disappeared.  A per-id ``asyncio.Lock`` now
  wraps each read-modify-write flow.  **Scope:** in-process only.
  Two processes writing to the same backing database still race;
  the multi-process fix (optimistic-version / row-lock check at
  the database layer) is tracked as a separate work item.

- Bumped minimum `pyyaml` requirement from `>=6.0` to `>=6.0.2` to
  exclude versions that lack cp312/cp313 wheels and fail to build
  from source against modern Cython.  Surfaced by the floor
  resolve step in the `dependency-update` workflow.

### Migration

- **Stored record shape for `DataKnobsRegistryAdapter` changed.**
  Pre-migration, every field of the `Registration` was written into
  the ``data`` column and the record's metadata column was always
  empty (there was no ``Registration.metadata`` field).
  Post-migration, `Registration.metadata` is written to the
  record's ``metadata`` column.  Existing deployments must rewrite
  stored rows once before the new `filter_metadata=` / metadata
  pushdown will see anything (the column is empty on pre-migration
  rows).

- **Wire-protocol change is additive.** `Registration.to_dict()`
  / `from_dict()` gained a ``metadata`` key.  Old clients that
  ignore unknown keys keep working against new servers; old
  servers that omit the key produce ``metadata={}`` on the new
  client via ``data.get("metadata") or {}``.  No coordinated
  upgrade is required, but until both sides understand the key,
  the metadata channel is effectively absent on that consumer.

- **New `ArtifactRegistry.query` parameters (`filter_metadata=`,
  `sort=`, `limit=`, `offset=`) are kw-only.**  This is the
  contract for the new surface; positional usage of the
  established parameters (`artifact_type`, `status`, `tags`,
  `filters`) is unchanged.

## v0.6.19 - 2026-05-09

### Added

- **`VectorMemory(immutable_metadata_keys=...)`** — declares which
  `default_metadata` keys cannot be overridden by caller-supplied
  `metadata` on `add_message()`. Use for tenant-scoping identifiers
  (e.g. `immutable_metadata_keys=["user_id"]` paired with
  `default_metadata={"user_id": "..."}`). Caller-attempted overrides
  are logged as warnings and the configured value is preserved.
  Plumbed through `VectorMemory.from_config()`.

- **`VectorMemory.clear(filter_metadata=...)`** — filter-aware
  clear. When called with no args on a `VectorMemory` constructed
  with `default_filter=...`, the default filter is auto-applied,
  making `mem.clear()` symmetric with `mem.get_context()` for
  tenant-scoped instances. Pass `filter_metadata=...` explicitly to
  scope a clear to a different subset (e.g. one
  category/conversation within a tenant).

- **`RAGKnowledgeBase.clear(filter=...)`** — filter-aware clear,
  passing through to the underlying `VectorStore.clear(filter=)`.

### Fixed

- **`RAGKnowledgeBase._embed_and_store_chunks` no longer lets
  caller `metadata` overwrite system-controlled chunk fields**
  (`text`, `source`, `chunk_index`, `document_type`,
  `source_path`). Pre-fix, an ingest call passing
  `metadata={"text": "tampered"}` could silently corrupt stored
  chunks; the bug was reachable through every public ingest entry
  point. Caller-supplied values for system fields are now logged
  as warnings via `dataknobs_common.metadata.enforce_immutable_keys`
  and the system value is preserved.

- **`KnowledgeIngestionManager.ingest(domain_id, clear_existing=True)`
  no longer wipes other domains' chunks in a shared vector store.**
  Pre-fix, the manager called the underlying `VectorStore.clear()`
  with no filter, so refreshing one domain in a multi-tenant store
  removed every other domain's chunks silently. Post-fix, the clear
  is scoped by `domain_id` via
  `RAGKnowledgeBase.clear(filter={"domain_id": domain_id})`.
  Consumer-side workarounds (e.g. defaulting `clear_first=False`
  to dodge the issue) can be reverted on upgrade.

- **`RAGKnowledgeBase._embed_and_store_chunks` chunk IDs are now
  scoped by `domain_id` when present in the threaded metadata.**
  Pre-fix, the chunk-id stem was derived purely from
  `Path(source_file).stem`, so two chunks at the same relative
  filename across different domains (e.g. `domain-a/doc.md` and
  `domain-b/doc.md`) collided on a shared store and the second
  ingest upserted over the first. Post-fix, the chunk-id prefix
  becomes `f"{domain_id}\x1f{stem}"` whenever `domain_id` is in the
  caller-supplied metadata (which `KnowledgeIngestionManager`
  threads automatically). The record-separator (`\x1f`) between
  `domain_id` and `stem` rules out snake_case-domain collisions
  (`my` + `team_doc` vs `my_team` + `doc` would otherwise both
  produce `my_team_doc` under `_`). Single-domain consumers
  (no `domain_id` threaded) see **no change** — chunk IDs keep the
  historical `f"{stem}_{index}"` form, so re-ingest into existing
  populated stores remains idempotent.

- **`RAGKnowledgeBase.ingest_from_backend` no longer threads the
  redundant `source` and `filename` keys** that
  `KnowledgeBaseConfig.get_metadata` adds, into
  `_embed_and_store_chunks`. The chunk-build step already receives
  the more-precise `source_file` (display URI) and `source_path`
  (relative path) explicitly; dropping the redundant copies stops
  the new immutable-key enforcer from emitting a spurious warning
  on every legitimate ingest.

### Changed

- **`VectorMemory.clear()` semantics on tenant-scoped instances.**
  When `default_filter` is set, `clear()` (no args) now removes
  only the matching tenant's vectors, not the entire store. The
  pre-fix unscoped behavior was a documented gap (Brief 118
  sub-issue 8b); the docs steered consumers away from production
  `clear()` because it could not respect tenant scoping. This is
  a behavior change for tenant-scoped instances — consumers who
  genuinely want to wipe all tenants from a shared store should
  call `mem.vector_store.clear()` directly (bypassing the
  `VectorMemory` wrapper).

- **`VectorMemory.clear(filter_metadata=...)` now AND-composes
  with `default_filter` instead of replacing it.** Pre-fix, an
  explicit `filter_metadata` argument took full precedence over
  the memory's `default_filter`, allowing a tenant-scoped instance
  to wipe other tenants' rows in a shared store via an explicit
  override (e.g. tenant-A's memory could call
  `clear(filter_metadata={"user_id": "B"})` and remove tenant B's
  data). Post-fix the filters AND-compose, so explicit filters
  narrow WITHIN the tenant scope and never escape it. On key
  collision (caller passes a key that conflicts with the default)
  the merged filter contains contradictory clauses and matches
  nothing — the clear is a no-op rather than a cross-tenant wipe.

- **`KnowledgeBase` ABC now declares `clear(filter=...)`** with a
  default `NotImplementedError`. `RAGKnowledgeBase` overrides it
  with the filter-aware delete path. Subclasses that don't support
  deletion get a clean error rather than being silently
  mis-driven by managers like `KnowledgeIngestionManager`.

### Fixed

- **`MarkdownChunker.ChunkMetadata.to_dict()` no longer lets
  `custom` overwrite structured fields.** Pre-fix, `to_dict` ended
  with `**self.custom`, so a custom entry sharing a key with a
  structured field (`headings`, `chunk_index`, `chunk_size`,
  etc.) silently overwrote the structured value in the serialized
  dict — same vulnerability class as the pre-118 `_create_chunk`
  `node_type` defense, but covering the entire system-field
  surface. Post-fix, `**self.custom` is unpacked first so
  structured fields win.

- **`RAGKnowledgeBase._embed_and_store_chunks` chunk-id separator
  switched from `_` to `\x1f` (ASCII unit separator)** to
  eliminate snake-case-domain collisions. Pre-fix, the
  underscore-joined prefix caused
  `domain_id="my"`+file `team_doc.md` to collide with
  `domain_id="my_team"`+file `doc.md` (both produced
  `my_team_doc_0`). The unit-separator character cannot appear in
  domain IDs or file stems, so collisions are structurally
  impossible. Chunk IDs are not part of any documented public
  surface, so this is a safe internal change.

- **`RAGKnowledgeBase._embed_and_store_chunks` strips redundant
  `source` / `filename` keys from caller metadata at the shared
  layer.** Pre-fix, the strip lived only in
  `ingest_from_backend`, so direct callers of
  `load_markdown_text(metadata={"source": "..."})` still
  triggered a spurious immutable-key warning even though the
  caller's `source` was a redundant copy of the explicit
  `source_file` argument (different views of the same file). Now
  every entry point benefits.

- **Immutable-key warnings are emitted once per offense, not once
  per chunk.** Pre-fix, the per-chunk loop in
  `_embed_and_store_chunks` invoked `enforce_immutable_keys` on
  every chunk, so an N-chunk document with one bad metadata blob
  emitted N identical warnings. Post-fix, the helper is invoked
  with `caller=metadata` on the first chunk only (warning
  emission) and `caller=None` on subsequent chunks (silent
  enforcement) — one warning per offense.

### Migration

- Callers who currently rely on `default_metadata` for tenant
  scoping should add `immutable_metadata_keys=[...]` matching the
  scoping keys. Existing callers who do not set
  `immutable_metadata_keys` see no behavior change for
  `add_message` — caller metadata still wins on every key (the
  pre-fix default).
- Callers who relied on `VectorMemory.clear(filter_metadata=...)`
  as a "broader" wipe than `default_filter` (e.g. a tenant-A memory
  passing `filter_metadata={"category": "X"}` expecting to wipe
  category X across ALL tenants in the shared store) must update
  their code: the explicit filter now narrows WITHIN the tenant
  scope. For an all-tenants wipe, drop down to the underlying
  vector store: `mem.vector_store.clear(filter={"category": "X"})`.
- Callers of `RAGKnowledgeBase` ingest methods who passed
  caller-`metadata` containing `text`/`source`/`chunk_index`/
  `document_type`/`source_path` (a bug-shaped pattern) must update
  their code: those keys are now system-controlled and caller
  values are logged as warnings and discarded.
- **`VectorMemory.clear()` on tenant-scoped instances now
  auto-applies `default_filter`.** Code that called `clear()` to
  wipe an entire shared store (regardless of tenant scoping) will
  now wipe only the calling tenant's slice. Consumers who meant
  the all-tenants wipe should call `mem.vector_store.clear()`
  directly.
- **`KnowledgeIngestionManager.ingest(clear_existing=True)` is now
  safe in shared stores.** Workarounds that flipped
  `clear_existing` to `False` to avoid cross-domain wipes can be
  reverted on upgrade.

### Security
- Bumped minimum `jinja2` requirement from `>=3.1.0` to `>=3.1.6`
  to exclude versions affected by GHSA-cpwx-vrp4-4pq7,
  GHSA-gmj6-6f8f-6699, GHSA-h75v-3vvj-5mfj, and GHSA-q2x7-8rv6-6q7h.

### Added
- `EnsureIngestionResult.duration_seconds` property — counterpart
  to `IngestionResult.duration_seconds`. Computes
  `completed_at - started_at` in seconds. Returns `float` (not
  `float | None`): `EnsureIngestionResult.completed_at` is typed
  as `datetime` with a construction-time default factory, so a
  terminal result's duration is always defined.
- `RegistryBackend.peek_config(bot_id)` — non-mutating sibling of
  `get_config`. Returns the stored config dict without updating
  `last_accessed_at`, for inspection / audit / bookkeeping reads
  that should not register as user activity. Implemented on
  `InMemoryBackend`, `DataKnobsRegistryAdapter`, and
  `HTTPRegistryBackend`. The HTTP backend has no client-side
  activity state, so its `peek_config` delegates to `get_config`;
  servers that want to distinguish a non-touching peek may define
  their own contract (header, query parameter, or sibling
  endpoint) — this client deliberately does not impose one.

### Changed
- `BotRegistry.get_config()` now routes through
  `RegistryBackend.peek_config` rather than `get_config`.
  Inspection-style reads no longer bump `last_accessed_at`;
  consumers needing the touching behavior should use
  `BotRegistry.get_bot()`, which is the user-facing resolution
  path.
- `BotRegistry.get_bot()` now touches the backend on every call
  (cache hit and miss alike) so `last_accessed_at` reliably
  reflects user activity. Previously the backend `get_config`
  was issued only on the cache-miss branch, which produced an
  inverted activity signal — hot bots (always cache hits) never
  updated, cold bots updated only on TTL expiry. The change adds
  one backend read per `get_bot` call; for the HTTP backend that
  is one extra round trip per call, for the
  `DataKnobsRegistryAdapter` it is one extra `db.read` plus the
  pre-existing `db.update` that `get_config` already performed.
- `ConfigCachingManager.get_raw_config()` now routes through
  `RegistryBackend.peek_config`. Bypassing the cache also bypasses
  the activity bump, matching the inspection-path role the method
  already documents.
- `CachingRegistryManager.get_or_create()` cache-miss reads now
  route through `RegistryBackend.peek_config`. Previously
  `last_accessed_at` was bumped only on cache misses (cache hits
  bypass the backend), producing an inverted activity signal —
  hot bots never updated, cold bots updated only on TTL expiry.
  Storage timestamps now reflect direct backend reads only;
  user-activity tracking for `CachingRegistryManager` consumers
  belongs at the `get_or_create` caller (or higher) — if your
  deployment relied on cache-miss-as-activity, call
  `backend.get_config()` directly in the request-handling path,
  or migrate the call site to `BotRegistry.get_bot()` (which now
  bumps unconditionally).
- Non-UTF-8 backend bytes for a knowledge-base config raise
  `IngestionConfigError` from
  `RAGKnowledgeBase._load_kb_config_from_backend`. Previously a
  stray `UnicodeDecodeError` could escape this path.
- `EnsureIngestionResult.completed_at` is typed as `datetime`
  (non-optional) with a construction-time default factory. Every
  terminal state — skip, error, success — produced by
  `KnowledgeIngestionService.ensure_ingested`,
  `KnowledgeIngestionService.ingest_from_config`, and
  `AutoIngestionMixin._ensure_knowledge_base_ingested` carries a
  real timestamp; consumers that serialize via `to_dict()` see a
  consistent `"completed_at"` on every result. The
  ``IngestionResult`` → ``EnsureIngestionResult`` boundary in
  `from_ingestion_result` coalesces a not-yet-completed source
  (`IngestionResult.completed_at is None`) to
  `datetime.now(timezone.utc)` rather than weakening the
  invariant.
- `EnsureIngestionResult.to_dict()` now serializes `started_at`
  (ISO format), `completed_at` (ISO format), and
  `duration_seconds` — bringing it into shape parity with
  `IngestionResult.to_dict()`. Strict superset of prior keys; no
  removed keys.

### Internal
- `RAGKnowledgeBase._load_kb_config_from_backend` uses
  `dataknobs_common.config_loading.parse_yaml_or_json` for the
  bytes → dict parse. Surface is `IngestionConfigError`.

## v0.6.18 - 2026-05-06

## v0.6.17 - 2026-04-29

### Added
- `RAGKnowledgeBase.ingest_from_backend(backend, domain_id,
  config=None, progress_callback=None, extra_metadata=None)` —
  unified ingest for any `KnowledgeResourceBackend` (file, memory,
  S3) with full `KnowledgeBaseConfig` support: patterns, exclude
  patterns, per-pattern chunking overrides, streaming JSON/JSONL.
  When `config` is `None`, auto-loads
  `knowledge_base.(yaml|yml|json)` from the domain root (falling
  back to `_metadata/knowledge_base.*`); a malformed config raises
  `IngestionConfigError`. `extra_metadata` is merged onto every
  chunk — `KnowledgeIngestionManager` uses this to thread
  `{"domain_id": domain_id}` onto chunks so multi-tenant queries
  can filter on it.
- `IngestOrchestrator` (`dataknobs_bots.knowledge.orchestration`) —
  subscriber-side primitive that listens on an `EventBus` trigger
  topic and dispatches to
  `KnowledgeIngestionManager.ingest_if_changed`. Concurrent triggers
  for the same `domain_id` are serialized via a per-domain
  `asyncio.Lock`; different domains proceed in parallel. Stateless
  across restarts; trigger adapters (S3/SQS/cron → bus) remain
  consumer responsibility.
- `BackendDocumentSource` (re-exported from
  `dataknobs_xization.ingestion`) — adapts any
  `KnowledgeResourceBackend` to the `DocumentSource` protocol.
  Derives a common literal prefix from configured patterns and
  passes it to `backend.list_files(prefix=...)` so S3 (and any
  other prefix-aware backend) can avoid listing the whole bucket.
- `KnowledgeIngestionManager.ingest_if_changed(domain_id,
  last_version=None)` returning `IngestionResult | None` —
  returns `None` (and skips the ingest) when `last_version` is
  supplied and the backend reports no changes.
- `S3KnowledgeBackend` accepts a pre-built
  `session_config: S3SessionConfig` kwarg for sharing a single S3
  configuration across multiple backends.

### Changed
- `KnowledgeIngestionManager.ingest()` delegates to
  `RAGKnowledgeBase.ingest_from_backend` and threads
  `{"domain_id": domain_id}` into per-chunk metadata so downstream
  queries can filter by tenant.
- `S3KnowledgeBackend` `region` default flipped from `"us-east-1"`
  to `None`; client routes through `create_boto3_s3_client`. See
  `dataknobs-data` notes above for the behavior-change details and
  migration guidance.
- `S3KnowledgeBackend.from_config` accepts both `region` and
  `region_name` keys (parity with `SyncS3Database` /
  `AsyncS3Database`).

# Changelog

All notable changes to the dataknobs-llm package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- **Token-budget truncation signal.** New `LLMResponse.truncated` /
  `LLMStreamResponse.truncated` boolean (default `False`) that every provider
  populates when generation is cut off at the token budget — Anthropic
  (`stop_reason == "max_tokens"`), OpenAI (`finish_reason == "length"`), Ollama
  (`done_reason == "length"`), and Bedrock (`stopReason == "max_tokens"`). A
  truncated response is incomplete; most dangerously a truncated tool-call turn
  carries partial arguments that look well-formed. A shared base
  `_warn_if_truncated()` hook (wired into `_analyze_response` and every
  streaming final-chunk assembly) logs a `warning` on a truncated tool-call
  turn and `info` on plain text, so the signal surfaces once and consistently
  across the `complete`, `stream_complete`, and deprecated `function_call`
  paths of every provider. HuggingFace's inference path exposes no stop-reason
  signal, so `truncated` stays `False` there.
- `normalize_claude_stop_reason()` in `dataknobs_llm.llm.base` — the shared
  Claude-family stop-reason normalizer used by both the Anthropic and Bedrock
  adapters (Bedrock runs Claude, so the two share the vocabulary verbatim).
- **Shared message-sequence utility** `dataknobs_llm.llm.message_sequence` — a
  provider-agnostic home for structural conversation-history invariants,
  expressed as pure functions over `list[LLMMessage]` (never mutating the
  input). Ships `pair_orphan_tool_calls()` (pair a dangling assistant
  `tool_use` with a synthetic `tool_result` so the request is valid on every
  backend — Anthropic 400s on a dangling `tool_use`) and `tool_call_signature()`
  (the canonical `(name, sorted-params-json)` duplicate-detection key, shared by
  the pairing repair and reasoning-loop duplicate-break guards so they agree by
  construction).
- **Configurable mid-conversation system-message policy for Anthropic** via
  `LLMConfig.options["system_message_policy"]`: `inline` (**default** — convert
  a mid-conversation `role="system"` message to a `user` message at its
  position, consolidating content blocks so the request stays valid: no
  consecutive same-role turns, and `tool_result` blocks kept first per
  Anthropic's ordering rule), `hoist` (legacy — merge into the top-level
  `system` param), `warn` (log then hoist), `reject` (raise `ValidationError`).
  An unknown policy fails closed at provider construction. Whether a family
  accepts an inline system message reads from the `ModelConstraints.accepts_inline_system`
  datum (`False` for Anthropic).
- **`max_tokens` is clamped to the model's output ceiling** (and rejected
  sampling params dropped) at a **shared, provider-agnostic choke point.** The
  S1 `ModelConstraints.max_tokens_ceiling` datum is wired through a new base
  `LLMProvider._apply_request_constraints()` that shapes the runtime config in
  canonical space (before any provider `adapt_config`): when a request's
  `max_tokens` exceeds the model's ceiling it clamps down to the ceiling
  (clamp-and-warn, never silent), and it drops any family-rejected sampling
  param (e.g. Claude 5's `temperature`). Because the shaping is provider-
  agnostic, the **same** clamp/drop serves both Claude providers — the native
  Anthropic Messages API **and** Amazon Bedrock Converse (Claude-on-Bedrock) —
  with no per-provider duplication; the shared Claude family knowledge (the
  bundled ceiling resource + the Claude-5 `temperature` rule) lives in
  `llm/providers/_claude_shared.py`. Clamping *down* is always a valid request,
  so this pre-empts the output-truncation / 400 class at source rather than
  recovering from it. On the native Anthropic endpoint the ceiling is **resolved
  dynamically** from the live Models API (`max_tokens`), cached per process and
  refreshed on a configurable TTL (`options["model_limits_ttl"]`, default
  `3600`s; at most one `models.list()` per TTL per event loop, never per
  request), with each poll independently bounded by
  `options["model_limits_refresh_timeout"]` (default `10`s) so a *hung* control
  plane cannot stall the request path. It falls back to the maintained bundled
  resource (`llm/providers/data/anthropic_model_limits.yaml`) — the same
  resource Bedrock uses (it has no Models API) — when the live API is
  unavailable, and a known-good dynamic value is never degraded back to the
  resource on a transient failure. The dynamic cache and the resource share one
  family-matching rule, so a bare-alias request resolves a dated cache/resource
  key and vice versa. Dynamic resolution can be disabled with
  `options["model_limits_dynamic"] = false` (resource-only) or forced with
  `await provider.refresh_model_limits()`; `initialize()` performs no network
  I/O — the ceiling is refreshed lazily at the first request boundary. Always
  config-overridable via `LLMConfig.constraints={"max_tokens_ceiling": N}`
  (per-field overlay, always wins over the dynamically-resolved value). The
  bundled resource is kept honest by a maintainer tool
  (`bin/update-model-limits.sh --check` / `--update`, key-gated). Additive and
  non-breaking: an unknown model resolves to `None` (permissive, unchanged), and
  the default `max_tokens` (`1024`) is below any real ceiling, so the
  overwhelming majority of requests are byte-identical.

### Changed

- **Anthropic mid-conversation `role="system"` messages default to `inline`
  (was silently hoisted).** A **leading** system prompt still always hoists into
  the top-level `system` param (unchanged). A **mid-conversation** system
  message now inlines at its position by default, preserving its in-context
  meaning instead of becoming a standing global instruction. Set
  `options["system_message_policy"] = "hoist"` to restore the exact legacy
  request shape byte-for-byte. In-tree histories carry only leading system
  messages (which hoist under every policy), so the change affects external
  consumers that emit mid-conversation system messages.

- **`finish_reason` is now the canonical vocabulary (`stop` / `length` /
  `tool_calls`) for every provider.** The Claude-family providers (Anthropic,
  Bedrock) normalize their raw stop reason onto it (`max_tokens` → `length`,
  `tool_use` → `tool_calls`, `end_turn`/`stop_sequence` → `stop`); OpenAI and
  Ollama already emitted it. The raw provider value is preserved on
  `metadata["raw_finish_reason"]` for any caller that needs the exact token.
- `LLMResponse.truncated` was inserted between the `finish_reason` and `usage`
  fields. All in-tree construction uses keyword arguments; any external code
  constructing `LLMResponse` **positionally** past `finish_reason` must add the
  `truncated` argument or switch to keywords.
- **Vendor API errors now raise `dataknobs_common.exceptions` types across
  every provider.** Anthropic, OpenAI, Ollama, HuggingFace, and Bedrock now
  translate transport errors uniformly: a 429 →
  `RateLimitError` (carrying `retry_after` when the provider exposes it), a
  400 → `ValidationError`, and auth / permission / connection / timeout / any
  other status → `OperationError`. The original SDK / transport error is
  preserved on `__cause__`, and a non-vendor exception (a bug in caller code)
  propagates unchanged rather than being masked as an API error. The
  status→type policy lives once on `LLMProvider._dataknobs_error_for_status`,
  and the raise / stream-iteration choke points (`_raise_translated`,
  `_call_api`, `_iter_translated`) are shared on the base too; each provider
  contributes only a small SDK-specific extractor. Translation covers every
  entry point — `complete`, `stream_complete`, `embed`, and the deprecated
  `function_call` — and on the streaming path a vendor error is translated
  whether it surfaces at stream creation or mid-iteration. `retry_after` is
  parsed from either form the `Retry-After` header permits (a number of
  seconds or an HTTP-date). Ollama's deprecated `function_call` falls back to
  prompt-based calling only for the genuine "model lacks the native tools API"
  `400` signal — a rate-limit / auth / transport error surfaces as its
  dataknobs exception instead of triggering a second request. **Backward
  compatibility:** any consumer that previously caught a *raw vendor type*
  around a provider call (`except openai.RateLimitError`, `except
  aiohttp.ClientResponseError`, `except botocore.exceptions.ClientError`, etc.)
  must now catch the corresponding `dataknobs_common.exceptions` type (the raw
  error remains reachable via `__cause__`). Domain-specific errors such as
  `ToolsNotSupportedError` (Ollama / HuggingFace) are unaffected — they are
  raised ahead of, and never flattened by, the translator.

## v0.6.7 - 2026-07-20

## v0.6.6 - 2026-07-15

### Security

- Bumped minimum `transformers` requirement from `>=5.3.0` to `>=5.5.0` to
  exclude versions affected by GHSA-fgcw-684q-jj6r / CVE-2026-5241 (CVSS 8.0,
  arbitrary code execution during LightGlue model initialization), fixed in
  5.5.0. Flagged at the floor resolve by the `dependency-update` workflow.
- Bumped minimum `torch` requirement from `>=2.12.0` to `>=2.13.0` to sweep the
  transitive `setuptools` floor past PYSEC-2026-3447 / CVE-2026-59890 (CVSS 6.1,
  `MANIFEST.in` glob path traversal in `setuptools.FileList`), fixed in
  setuptools 83.0.0. torch 2.12.0 pins `setuptools<82` (floor-resolving to the
  vulnerable 81.0.0); torch>=2.13.0 requires `setuptools>=77.0.3` with no upper
  cap, letting the floor resolve reach the fixed 83.0.0. The still-unfixed
  GHSA-rrmf-rvhw-rf47 (`torch.jit.script`, not called from this codebase)
  remains accepted; the inline floor comment in `pyproject.toml` records the
  rationale.

## v0.6.5 - 2026-07-07

### Added

- `BedrockProvider` — an Amazon Bedrock LLM provider registered as
  `"bedrock"`, serving **both** chat/completion (via the unified Converse
  API) and embeddings (Amazon Titan / Cohere via `invoke_model`) from a
  single provider. Authentication is via the AWS credential chain (IAM
  role, environment, or shared config) — there is no API key; region,
  endpoint, explicit credentials, and Bedrock guardrail settings are
  supplied through `LLMConfig.options`. Streaming, tool use, and
  cross-region inference-profile model ids are supported. The provider
  reuses the shared, loop-safe `dataknobs_common.aws.create_aioboto3_session`
  factory (warmed for `bedrock-runtime`), so session construction never
  blocks the event loop, and opens its per-operation `bedrock-runtime`
  clients through the shared `AwsSessionConfig.to_session_client_kwargs()`
  builder — so every `complete` / `function_call` / embed call carries an
  explicit socket read timeout (`LLMConfig.timeout`, default `60`s), retry /
  connection-pool tuning, and the `endpoint_url` / `use_ssl` handling shared
  with every other AWS consumer. `stream_complete` decouples its per-read
  (inter-chunk) timeout from the whole-response budget via the
  `stream_read_timeout` option (default: boto's `60`s), since applying the
  total `timeout` as a per-read timeout would kill a stream on a long
  inter-token pause. Embedding knobs are configurable via `options`:
  `normalize` (Titan), `input_type` (Cohere — `search_query` at query time),
  and `embed_max_concurrency` (bounds Titan's per-text `invoke_model` fan-out
  so a large batch cannot trip throttling; defaults to
  `max_pool_connections`). Invalid numeric options (e.g. a non-integer
  `embed_max_concurrency`) raise `ConfigurationError` naming the option.
  Capability detection reports embedding-only models as `EMBEDDINGS` only
  (they no longer advertise chat / streaming / text-generation). Partial
  explicit AWS credentials fail closed at construction. `BedrockProvider`
  is exported from the package root; `BedrockConverseAdapter` from
  `dataknobs_llm.llm`. Install the async transport with
  `pip install 'dataknobs-llm[bedrock]'` (composes
  `dataknobs-common[aws]`; `aioboto3` is lazy-imported, so the base install
  is unaffected).

### Security

- Bumped minimum `transformers` requirement (extra: `embeddings`) from
  `>=5.0.0` to `>=5.3.0` to exclude GHSA-29pf-2h5f-8g72 (CVSS 7.8),
  fixed in 5.3.0 and flagged at the floor resolve by the
  `dependency-update` workflow. The bump preserves the prior
  GHSA-69w3-r845-3855 (CVSS 6.5) floor and the earlier CVE sweep
  (highest CVSS 9.0: PYSEC-2023-300). The inline floor comment in
  `pyproject.toml` records the rationale.

## v0.6.4 - 2026-06-23

## v0.6.3 - 2026-06-22

### Added

- `ExecutionTracker` composes an in-process `CallbackRegistry`. Every
  `record(...)` / `record_async(...)` fires the `execution:record` topic
  (`EXECUTION_RECORD_TOPIC`) on the lazily constructed
  `execution_callbacks` registry with a
  `{tool_name, success, duration_ms, error}` payload. For cross-replica
  fan-out, compose `execution_callbacks.also_publish_to(...)` and drive
  recording through `record_async` — it fires via `fire_async`, so bus
  delivery is awaited correctly from inside a running event loop.
  `ToolRegistry.execute_tool` records via `record_async`, so tracked
  tool execution gets fan-out for free; sync `record(...)` with fan-out
  composed inside a running loop is rejected (use `record_async`). The
  existing `record / query / get_stats / clear / __len__` surface is
  unchanged. Advertises `Capability.EXECUTION_TRACKING` /
  `CALLBACK_REGISTRY`.
- `dataknobs_llm.intent` module — pluggable intent-classification
  surface for any LLM-layer consumer that needs to route user input
  by intent (tool routers, reasoning strategies, RAG query
  classifiers, downstream packages with wizard or routing flows that
  consume an `intent_detection:` block). `IntentClassifier` is a small
  `@runtime_checkable` Protocol with one async
  `classify(message, intents, **kwargs) -> IntentMatchResult`
  method. `IntentSpec` and `IntentMatchResult` are frozen
  dataclasses; `IntentMatchResult.confidence: float | None` is
  reserved for future calibrated-confidence classifiers (the
  built-in keyword / JSON-output LLM / composite / negation-filter
  classifiers return `None`).
- `KeywordIntentClassifier` — rule-based classifier with injectable
  `vocabulary` and `tokenizer`. Default tokenizer is
  `default_word_boundary_tokenizer` (word-boundary regex): a bare
  vocabulary entry `"yes"` matches a standalone `"yes"` but not the
  `"yes"` substring of `"yesterday"`. Inject a custom tokenizer for
  I18N / fuzzy / N-gram / morphological matching.
- `LLMIntentClassifier` — LLM-backed classifier with injectable
  `llm: AsyncLLMProvider | None` and `prompt_template`. Lenient
  response parsing accepts both the `DEFAULT_LLM_PROMPT_TEMPLATE`
  JSON shape (`{"intent": ..., "extracted": ...}`) and a bare
  intent ID matched against the configured intent names. The
  extracted payload is coerced to the documented `str | None` shape
  (single-element list / number / bool is coerced; multi-element
  list / dict drops to `None`). Prompt intent-list ordering follows
  caller order rather than set-iteration order for prompt-cache hit
  rate and LLM-eval reproducibility. Provider errors are absorbed
  with a warning so an LLM outage returns no-match rather than
  crashing the caller; `asyncio.CancelledError` propagates.
- `CompositeIntentClassifier` — chains backends with
  `"first_match"` (default) or `"vote"` strategies. First-match is
  the standard "keyword first, optional LLM fallback" shape:
  `CompositeIntentClassifier([KeywordIntentClassifier(),
  LLMIntentClassifier()])`. Vote queries every backend and breaks
  ties by classifier order. Construction with an empty
  classifier list raises `ValueError`.
- `NegationFilter` — decorator wrapping any `IntentClassifier` to
  drop matches when
  `dataknobs_llm.extraction.grounding.has_negation` fires on the
  message. Constructor takes `negation_keywords` (defaults to
  `DEFAULT_NEGATION_KEYWORDS`) and an optional `suppress_intents`
  whitelist (`None` suppresses all matches under negation). A
  `NegationFilter`-suppressed match carries `rule_based=False`
  (suppression is post-classify, not a rule match in its own
  right).
- `intent_classifier_backends` —
  `PluginRegistry[IntentClassifier]` in `dataknobs_llm.intent`
  mirroring the shape of
  `dataknobs_common.events.event_bus_backends`,
  `dataknobs_common.locks.lock_backends`, and
  `dataknobs_common.ratelimit.rate_limiter_backends`. Built-in
  factories (`"keyword"`, `"llm"`, `"composite"`) auto-register at
  import; consumers register their own backends (embedding
  similarity, fuzzy match, locale-specific keyword variants) under
  any name. The registry is parametrized with
  `validate_type=IntentClassifier` so an out-of-tree factory
  returning a non-conforming instance fails at `create()` time
  rather than at first use; `not_found_kind="intent_classifier"` +
  `not_found_exception=ValueError` preserves a plain `ValueError`
  (not the `NotFoundError` default) on unknown name. Conforms to
  `BackendRegistry` for `isinstance` checks. `IntentClassifierFactory`
  typealias preserved.
- `create_intent_classifier(name, config=None)` factory — resolves
  a registered backend by name through
  `intent_classifier_backends.create(key=name, config=...)` (the
  explicit-key mode of `PluginRegistry.create`, since no
  `config_key` is configured). Raises `ValueError` listing every
  registered backend on unknown name; the message shape is
  `Unknown intent_classifier: <name>. Available backends:
  <sorted-keys>` — same shape produced by `create_event_bus` /
  `create_lock` / `create_rate_limiter` for their respective kinds.
  Factory failures (invalid config, missing required fields, etc.)
  are wrapped in `OperationError` with the originating exception
  preserved on `__cause__`. Composite child specs are themselves
  `{"classifier": <name>, "config": {...}}` mappings; a child
  missing the `classifier:` discriminator raises rather than
  silently dropping.
- `create_intent_classifier_async(name, config=None)` — async
  counterpart to `create_intent_classifier` that dispatches via
  `intent_classifier_backends.create_async(...)` so an out-of-tree
  classifier whose factory exposes `from_config_async` (or returns
  an awaitable) is detected and awaited. Built-in classifiers
  construct synchronously; the async shim returns the same
  instance type as the sync shim for identical input.
- Package-root re-exports for `dataknobs_llm.intent`:
  `DEFAULT_VOCABULARY`, `DEFAULT_LLM_PROMPT_TEMPLATE`,
  `DEFAULT_NEGATION_KEYWORDS`, `DEFAULT_AFFIRMATIVE_SIGNALS`,
  `DEFAULT_NEGATIVE_SIGNALS`, `word_in_text`, and
  `default_word_boundary_tokenizer`. The single-token English
  yes/no vocabularies live in `dataknobs_llm.intent.defaults` under
  these public names; downstream consumers needing the same
  primitives for boolean recovery or analogous text-classification
  tasks import them from here directly.
- `KeywordIntentClassifier` `phrase_priority` mode — keyword-only
  constructor kwargs `phrase_priority: bool = False` and
  `phrases: Mapping[str, frozenset[str]] | None = None`. When opted
  in, multi-word phrase matches beat single-word matches; two
  intents tying at the same tier (both phrase-matched or both
  word-matched only) resolve to
  `IntentMatchResult(intent=None, ...)` rather than iteration-order
  first-match-wins. Default off — every call site without the opt-in
  keeps the first-match-wins iteration semantic.
  `dataknobs_llm.extraction.grounding.detect_boolean_signal` opts in
  so multi-word affirmative/negative phrases beat single-word
  matches; its public `bool | None` verdict is unchanged.

### Fixed

- **`SqliteEmbeddingCache.initialize` no longer blocks the event loop
  creating the cache directory.** It created the database's parent
  directory with a synchronous `mkdir` on the running loop; the `mkdir` is
  now offloaded via `asyncio.to_thread`. Behavior is unchanged.

### Security

- Acknowledged GHSA-rrmf-rvhw-rf47 (CVSS 5.3, local memory corruption
  via `torch.jit.script`) against the `torch>=2.12.0` floor (extra:
  `embeddings`), flagged at the floor resolve by the
  `dependency-update` workflow. The advisory affects all versions
  through 2.12.0 with no upstream fix. Risk accepted:
  `torch.jit.script` is not called from this codebase — `torch`
  enters only via the `transformers` embeddings extra, which uses
  eager-mode execution. The inline floor comment in `pyproject.toml`
  records the same rationale so future audits surface the accepted
  state rather than re-triaging the finding.

- Bumped minimum `aiohttp` requirement (extras: `ollama`,
  `huggingface`) from `>=3.13.4` to `>=3.14.1` to extend the prior
  `<=3.13.3` CVE sweep (highest CVSS 9.1: GHSA-63hf-3vf5-4wqf)
  through the full `<3.14.x` floor-resolve advisory set. The two
  named highs are GHSA-hg6j-4rv6-33pg (CVSS 6.6, cross-origin
  redirect cookie leakage on the per-request `cookies=` kwarg) and
  GHSA-jg22-mg44-37j8 (CVSS 6.4, `CookieJar.load()` deserialization);
  both were already triaged unreachable from this codebase (outbound
  HTTP uses header-based auth, the advisory's safe pattern, and
  `CookieJar.load()` is never invoked) but bumping clears the
  floor-resolve audit regardless. Fixes land across 3.14.0 and
  3.14.1, hence `>=3.14.1` as the floor. The bump was previously
  blocked by `aioresponses 0.7.8` not passing the `stream_writer`
  kwarg to `aiohttp.ClientResponse` introduced in aiohttp 3.14;
  unblocked by the workspace move off `aioresponses` to an
  in-process `aiohttp.web` test server in the bots package.

## v0.6.2 - 2026-06-06

### Added

- **Seed-aware metadata API on `ConversationManager`**
  (`dataknobs_llm.conversations`). `ConversationManager` carries
  metadata in two buckets: the live `state.metadata` (the unit of
  persistence) and an internal initial-metadata seed bucket. On the
  first `add_message`, the seed bucket is passed *by reference* into
  `ConversationState.metadata`, so post-first-materialization the two
  attributes name the same dict. `resume()` aliases the seed bucket to
  the loaded `state.metadata` so post-resume has the same shape — the
  two-bucket model is a pre-state distinction only; post-state the
  buckets are the same dict object. The existing
  `set_metadata` / `update_metadata` / `remove_metadata` family writes
  only to the live bucket, so it silently no-ops pre-state — by
  design, paired with the post-state-only `metadata` property
  (whose own pre-state return is `{}`). The new
  `seed_metadata(key, value)` / `update_seed_metadata(updates)` /
  `remove_seed_metadata(key)` / `get_seed_metadata(key=None)` family
  crosses the pre-/post-state boundary: pre-state the writers touch
  the seed bucket (the only bucket that exists), and post-state they
  touch the shared dict once. `await add_seed_metadata(key, value)` is
  the async, persisting analogue of `add_metadata` — pre-state it
  writes the seed bucket without raising, post-state it writes and
  immediately persists via `save()`. The `_writable_buckets()` /
  `_readable_bucket()` private generator helpers name the two-bucket
  abstraction once so the five public methods share one shape. The
  existing metadata methods are unchanged; each carries a `See Also:`
  pointer to its seed sibling so the gap is discoverable from the
  existing surface. None of the sync seed-* writers auto-persist —
  they match the existing sync non-persisting contract.
- **Public `ConversationManager.save()`** — durably persists the
  current state to storage. The metadata-method docstrings (existing
  AND seed-aware) already referenced `save()` as the public escape
  hatch for persisting sync writes; the method now exists. Delegates
  to the pre-existing private `_save_state()`. Silent no-op pre-state
  (nothing to persist).

### Fixed

- **`get_metadata` / `get_seed_metadata` now reject orphan `default`**.
  Pre-fix, `get_metadata(default="x")` (no `key`) silently discarded
  the default and returned the whole bucket dict — a quirk inherited
  from `dict.get` but ambiguous here because `key` is `Optional`. A
  consumer writing `manager.get_seed_metadata(default={"fallback": True})`
  (thinking "give me the bucket, or this fallback if empty") got `{}`,
  not the fallback. Passing `default` without `key` now raises
  `TypeError`. The normal `(key, default)` shape is unchanged. Pre-fix
  callers passing orphan `default` were silently buggy; the strict
  contract surfaces them at the call site.
- **`ConversationManager.resume()` now aliases the seed bucket to the
  loaded `state.metadata`**. Pre-fix, a resumed manager carried two
  divergent dicts — `_initial_metadata` was empty `{}` while
  `state.metadata` carried the loaded data. A post-resume
  `seed_metadata` write reached the empty seed bucket, but that
  bucket was never consumed again, so the write was effectively dead
  on the resume path. The alias makes the two attributes name the
  same dict object, matching the post-first-materialization shape —
  the two-bucket model collapses post-state across the entire
  lifecycle. No public-API change; consumers that read `state.metadata`
  or `get_seed_metadata()` see the same value before and after the
  fix when only `state.metadata` was populated, but
  `seed_metadata`/`update_seed_metadata`/`add_seed_metadata` writes
  on a resumed manager are now operationally meaningful.
- **`ToolRegistry.execute_tool` now forwards `_`-prefixed internal
  params to tools that accept `**kwargs`**. The method's docstring
  promised `_context` was passed to the tool but excluded from
  execution records; the implementation only honoured the
  exclusion-from-records half, silently stripping internal params
  before calling `tool.execute`. A `ContextAwareTool` invoked through
  the registry ran with the empty fallback context. The fix inspects
  the tool's `execute` signature: tools declaring `**kwargs` (chiefly
  `ContextAwareTool`) receive forwarded internal params; plain tools
  whose signatures don't accept `**kwargs` continue to receive only
  the non-`_` params (forwarding would otherwise raise `TypeError`).
  Records continue to exclude `_`-prefixed params, preserving the
  existing observability contract.
- **`ToolRegistry.execute_tool`'s tool-name parameter is now
  positional-only.** Pre-fix the signature was
  `execute_tool(self, name, **kwargs)`, so a tool whose parameters
  dict carried a `"name"` key (extremely common — user names, file
  names, target names) would collide with the positional `name` and
  raise `TypeError: got multiple values for argument 'name'`. The
  `/` positional-only marker lets `kwargs` freely include `name`.
  Surfaced when DynaBot routed its tool dispatch through the
  registry; the existing test suite did not exercise it because the
  pre-existing call sites used tool params like `operation` / `a` /
  `b` / `query` that didn't collide.

## v0.6.1 - 2026-06-02

### Added

- **History-redaction primitive** (`dataknobs_llm.conversations`):
  `HistoryRedaction` is a frozen `StructuredConfig` of
  `pattern` + `replacement`, eagerly compiled at construction so an
  empty `pattern` raises `ValueError` and an invalid regex raises
  `re.error` — both at config-load.
  `compile_history_redactions(redactions)` harvests the cached compiled
  patterns into `(compiled_pattern, replacement)` tuples for hot-path
  reuse, and `apply_history_redactions(messages, patterns, *, role_of,
  content_of, replace_content, redact_roles=frozenset({"assistant"}))`
  is shape-generic over an accessor trio so callers drive one
  implementation for any element shape — an `LLMMessage` here, a plain
  dict in `dataknobs-bots` memory backends.
  `apply_history_redactions_to_dicts` is the dict-shape convenience
  wrapper. Non-eligible-role elements pass through by identity (no
  shallow copy).
- **`HistoryRedactionMiddleware`** (`dataknobs_llm.conversations`).
  New `ConversationMiddleware` that rewrites assistant-role message
  content in `process_request` before it reaches the provider;
  `process_response` is a passthrough, so the fresh LLM response keeps
  its full citation set for rendering. Persisted conversation-tree
  nodes are never mutated — redaction is scoped to the in-memory
  message list this turn forwards to the LLM. Constructor accepts
  either a sequence of typed `HistoryRedaction` instances (the
  preferred shape — reuses the list a memory backend already carries)
  or the legacy ordered list of `{"pattern": <regex>, "replacement":
  <str>}` dicts; mixing the two in one call raises `TypeError`. Each
  dict spec is validated up front (missing `pattern` key or empty
  pattern raises `ValueError`). An optional `redact_roles=`
  defaults to `("assistant",)`. Non-content fields on the rewritten
  assistant message — `tool_calls`, `tool_call_id`, `name`,
  `function_call`, `metadata` — are preserved across the rewrite, so
  agent / tool-use loops keep their invocation and pairing fields
  intact. Patterns are applied in declared order: list the more
  specific pattern (a bracketed citation header) before the more
  general bare token, or the bare-token rule will consume the token
  inside the bracket and leave a malformed header.

### Security

- Bumped minimum `torch` requirement (extra: `embeddings`) from
  `>=2.9.0` to `>=2.12.0` to exclude PYSEC-2026-139 (CVSS 7.8,
  deserialization in the pt2 Loading Handler), flagged at the floor
  resolve by the `dependency-update` workflow. The OSV record's
  `last_affected: 2.10.0` makes 2.11.0+ unaffected per OSV semantics;
  2.12.0 was chosen as the latest stable. The bump preserves the
  prior sweep of PYSEC-2025-203/204/206 (fixed in 2.9.0),
  GHSA-887c-mr87-cxwp (CVSS 4.8, 2.8.0), GHSA-3749-ghw9-m3mg (CVSS
  3.3, 2.7.1), and CVE-2025-32434 (RCE in `torch.load`, 2.6.0).

## v0.6.0 - 2026-05-26

### Changed

- `LLMConfig` is now a frozen `StructuredConfig` (was a plain mutable
  dataclass). Fields can no longer be reassigned after construction — derive
  a varied config with `clone(**overrides)` instead. `from_dict` / `to_dict`
  are now inherited from the base.
  - `to_dict()` now emits **every** field, with unset optionals serialized as
    `None` (and `options` as `{}`), so that `from_dict(to_dict())` round-trips
    exactly. The previous hand-rolled `to_dict()` omitted `None`-valued fields;
    code that relied on those keys being absent must adjust. For a
    JSON-serialisable projection (enums rendered as their `.value`), use
    `to_json_dict()`.
  - `repr(config)` now masks `api_key` as `'***'` so the credential cannot leak
    to logs via `repr()` or an f-string. The stored value is unchanged and
    `to_dict()` still carries it for round-tripping.

### Added

- An `llm` resolver is registered into `config_registries`, so a raw `llm`
  config section (e.g. a bot's provider section) can be validated via
  `StructuredConfig.validate()` without constructing a provider.

### Security

- Bumped minimum `torch` requirement (extra: `embeddings`) from
  `>=2.8.0` to `>=2.9.0` to exclude PYSEC-2025-203 (CVSS 7.5),
  PYSEC-2025-204 (CVSS 7.5), and PYSEC-2025-206 (CVSS 5.3), flagged at
  the floor resolve by the `dependency-update` workflow. The bump
  preserves the prior sweep of GHSA-887c-mr87-cxwp (CVSS 4.8, 2.8.0),
  GHSA-3749-ghw9-m3mg (CVSS 3.3, 2.7.1), and CVE-2025-32434 (RCE in
  `torch.load`, 2.6.0). PYSEC-2026-139 (CVSS 7.8) has no upstream fix
  yet and remains flagged; it will be addressed when a fixed release
  ships.

## v0.5.14 - 2026-05-20

## v0.5.13 - 2026-05-18

## v0.5.12 - 2026-05-13

### Security
- Bumped minimum `transformers` requirement (extra: `embeddings`) from
  `>=4.53.0` to `>=5.0.0` to exclude GHSA-69w3-r845-3855 (CVSS 6.5),
  the first CVE not covered by the prior floor. 5.0.0 is the GA release
  fixing the new issue. Verified locally via `bin/dk pr --all` — the
  three transformers usage sites in
  `fsm_integration/resources.py` (`pipeline`, `AutoTokenizer`,
  `AutoModel`) are stable across the 4.x → 5.x boundary.
- Bumped minimum `torch` requirement (extra: `embeddings`) from
  `>=2.6.0` to `>=2.8.0` to exclude GHSA-887c-mr87-cxwp (CVSS 4.8,
  fixed in 2.8.0). The bump also sweeps GHSA-3749-ghw9-m3mg (CVSS 3.3,
  fixed in 2.7.1) and CVE-2025-32434 (RCE in `torch.load`, fixed in
  2.6.0). 2.8.0 was previously deferred for GA wheel coverage; coverage
  is now in place across supported platforms.

### Fixed
- Bumped minimum `pyyaml` requirement from `>=6.0` to `>=6.0.2` to
  exclude versions that lack cp312/cp313 wheels and fail to build from
  source against modern Cython (`'build_ext' object has no attribute
  'cython_sources'`). Surfaced by the floor resolve step in the
  `dependency-update` workflow.

## v0.5.11 - 2026-05-09

### Security
- Bumped minimum `aiohttp` requirement (extras: `ollama`, `huggingface`)
  from `>=3.8.0` to `>=3.13.4` to exclude 22 known CVEs (highest
  CVSS 9.1: GHSA-63hf-3vf5-4wqf), including CVE-2024-23334 / GHSA-5m98-qgg9-wh84.
- Bumped minimum `transformers` requirement (extra: `embeddings`) from
  `>=4.30.0` to `>=4.53.0` to exclude 16 known CVEs (highest CVSS 9.0:
  PYSEC-2023-300).
- Bumped minimum `jinja2` requirement from `>=3.1.0` to `>=3.1.6` to
  exclude versions affected by GHSA-cpwx-vrp4-4pq7, GHSA-gmj6-6f8f-6699,
  GHSA-h75v-3vvj-5mfj, and GHSA-q2x7-8rv6-6q7h.
- `torch>=2.6.0` (extra: `embeddings`) is unchanged. Two newer CVEs at
  CVSS 3.3 / 4.8 are tracked but the fix versions are 2.7.1-rc1 (not
  GA) / 2.8.0; will be revisited via the weekly CVE-audit workflow once
  GA wheels are available across supported platforms.

### Internal
- `FileSystemPromptLibrary._load_file` uses
  `dataknobs_common.config_loading.load_yaml_or_json`. Surface is
  `ValueError` for unsupported extensions, parse failures, and read
  errors. Empty / falsy parsed payloads collapse to `{}`.

## v0.5.10 - 2026-05-06

### Execution Layer

- `ParallelLLMExecutor` gains an opt-in `fail_fast` mode (default `False`,
  no behavior change for existing consumers). When enabled at the executor
  level (`__init__(fail_fast=True)`) or per call (`execute(...,
  fail_fast=True)` / `execute_mixed(..., fail_fast=True)` /
  `execute_sequential(..., fail_fast=True)`), the executor cancels
  remaining pending tasks on the first task failure. Cancelled tasks
  return `TaskResult(success=False, error=asyncio.CancelledError(...))`,
  distinguishable from completion-failures by the error type. Under
  `execute_sequential` the loop breaks on the first failure and the
  returned list is shorter than the input list (callers can detect
  short-circuit via `len(results) < len(tasks)`).
- `ParallelLLMExecutor` accepts `default_per_task_timeout`; `LLMTask` and
  `DeterministicTask` accept a per-task `timeout` override. When set,
  each task's body is bounded by `asyncio.wait_for`, returning
  `TaskResult(success=False, error=asyncio.TimeoutError(...))` on
  overrun. With `RetryConfig`, the timeout bounds each retry attempt
  individually (total elapsed across retries remains the consumer's
  responsibility). Sync `DeterministicTask` callables run on the thread
  executor and cannot be pre-empted mid-call; the awaiter stops waiting
  but the underlying thread continues until the function returns.

## v0.5.9 - 2026-04-29

### Test Infrastructure
- Postgres integration fixtures and the `test_storage_postgres.py` asyncpg
  call site now validate interpolated SQL identifiers via
  `dataknobs_common.testing.safe_sql_ident` (regex-validated; raises
  `ValueError` on anything outside `[A-Za-z_][A-Za-z0-9_]*`). The data-package
  conftest's `pg_database` lookup also moved from f-string interpolation to
  psycopg2 `%s` parameter binding for that string-literal site. Closes R1-01.

### Fixed
- `DataknobsConversationStorage` now propagates `state.metadata` into
  `Record.metadata` when persisting conversations. SQL backends with a
  dedicated metadata column (Postgres, Elasticsearch, etc.) can now
  index and query conversation metadata via
  `list_conversations(filter_metadata={...})` and
  `count_conversations(filter_metadata=...)`. Previously the metadata
  column was `NULL` on every conversation row and `metadata.<key>`
  filters returned no matches on those backends; in-memory backend
  behaviour is unchanged.

  Pre-fix rows in production Postgres databases remain queryable via
  `data->'metadata'`. To make pre-fix rows visible to `filter_metadata`
  on Postgres, run the following one-shot backfill (idempotent):

  ```sql
  UPDATE conversations
     SET metadata = data->'metadata'
   WHERE metadata IS NULL AND data ? 'metadata';
  ```

  (Substitute the actual table name if it isn't `conversations`.)

  Rows where `state.metadata` is an empty dict at save time have their
  metadata column set to `'{}'::jsonb`, not `NULL`. This is functionally
  equivalent to `NULL` for `filter_metadata` queries (no key matches an
  empty object) and matches the `Record.metadata` contract — no
  additional `WHERE` guard is needed on the consumer side.

  `state.metadata` is typed `Dict[str, Any]` and may contain
  JSON-serializable nested values (lists, dicts, numbers, booleans,
  strings, `None`); the in-tree wizard FSM persists nested state under
  `state.metadata["wizard"]`, and rate-limit/timing middleware write
  non-string scalars. On save, `_state_to_record` deep-copies
  `state.metadata` into `Record.metadata`, so post-save mutations of
  nested values do not leak into already-persisted rows. SQL backends
  with a dedicated metadata column index top-level keys;
  `filter_metadata={"key": value}` performs equality on the top-level
  value at that key, so nested-value filtering is outside the
  `filter_metadata` contract.

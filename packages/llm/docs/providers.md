# Provider Creation

Factory functions for creating and initializing LLM providers from configuration.

## Overview

The `dataknobs-llm` package provides three factory functions for creating providers:

| Function | Purpose |
|----------|---------|
| `create_llm_provider()` | Create a chat/completion provider |
| `create_embedding_provider()` | Create an embedding provider (initialized, mode forced) |
| `create_text_embedder()` | Create an embedder carrying its own `dimensions` and `model_id` |

All three use `LLMProviderFactory` internally and support all registered
provider backends (Ollama, OpenAI, Anthropic, Amazon Bedrock, HuggingFace,
Echo).

## create_llm_provider()

Create a chat/completion provider from configuration.

```python
from dataknobs_llm import create_llm_provider

provider = create_llm_provider({
    "provider": "ollama",
    "model": "llama3.2",
})

# Use the provider
response = await provider.complete(messages)
```

Accepts `LLMConfig`, `Config`, or `dict`. Returns an uninitialized provider —
call `await provider.initialize()` before use.

### Which of the two entry points to use

`create_llm_provider()` and `LLMProviderFactory(is_async=...).create()` build
the same object. What differs is where `is_async` lives, and that decides what
a type checker can tell you:

| Call | Returns |
|---|---|
| `create_llm_provider(config)` | `AsyncLLMProvider` |
| `create_llm_provider(config, is_async=False)` | `SyncProviderAdapter` |
| `create_llm_provider(config, is_async=some_bool)` | either — the union |
| `LLMProviderFactory(is_async=...).create(config)` | either — the union |

On the function, `is_async` is an argument, so overloads resolve the return
type to the one provider the call can actually produce. On the factory it is a
*constructor* flag, and `create()` has to stay callable through the `Config`
factory protocol — where the caller holds a factory object and not the flag
that built it — so it returns the union whatever the flag was set to.

Reach for the factory when the mode genuinely is not known at the call site,
or when registering it as a config factory. Otherwise prefer the function: a
caller that narrows the union with a check that cannot fail, or that assigns it
to something typed `Any`, is paying for an arm it can never receive.

The sync arm is `SyncProviderAdapter`, not `SyncLLMProvider`. The adapter wraps
an async provider rather than subclassing `LLMProvider`, and no
`SyncLLMProvider` subclass exists in tree — so `initialize()` and `close()` are
synchronous on that half and awaited only on the async one.

### Passing constructor arguments

`LLMProviderFactory.create()` forwards `**kwargs` to the provider constructor.
Every built-in provider takes `(config, prompt_builder=None)`; `EchoProvider`
takes several more:

```python
from dataknobs_llm import LLMProviderFactory

provider = LLMProviderFactory().create(
    {"provider": "echo", "model": "test"},
    responses=["a scripted reply"],
)
```

## create_embedding_provider()

Create and initialize an embedding provider from configuration. The provider is
returned ready for `embed()` calls with `CompletionMode.EMBEDDING` forced.

### Signature

```python
async def create_embedding_provider(
    config: LLMConfig | dict[str, Any],
    *,
    default_provider: str = "ollama",
    default_model: str = "nomic-embed-text",
) -> AsyncLLMProvider:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `LLMConfig \| dict` | — | A typed `LLMConfig` or a configuration dict (see formats below) |
| `default_provider` | `str` | `"ollama"` | Provider when not specified (dict path only) |
| `default_model` | `str` | `"nomic-embed-text"` | Model when not specified (dict path only) |

### Returns

Initialized `AsyncLLMProvider` with `CompletionMode.EMBEDDING` set.

### Configuration Formats

A typed `LLMConfig` or one of two dict formats is supported. An embedder
config **is** an `LLMConfig` — embedding providers ride the same provider
registry — so no separate config type is needed.

**Typed `LLMConfig`:**

```python
from dataknobs_llm import LLMConfig

provider = await create_embedding_provider(
    LLMConfig(provider="ollama", model="nomic-embed-text", dimensions=768)
)
```

`provider` / `model` are validated as required fields, and `mode` is forced to
`CompletionMode.EMBEDDING` (via `clone()` — `LLMConfig` is frozen, so the
caller's config is never mutated). `default_provider` / `default_model` are
unused on this path.

The two dict formats follow. The nested format is preferred.

**Nested format** (preferred):

```python
provider = await create_embedding_provider({
    "embedding": {
        "provider": "ollama",
        "model": "nomic-embed-text",
        "dimensions": 768,
        "api_base": "http://localhost:11434",
    },
})
```

All keys in the `embedding` sub-dict other than `provider` and `model` are
forwarded to the provider (e.g., `api_base`, `api_key`, `dimensions`).

**Legacy prefix format:**

```python
provider = await create_embedding_provider({
    "embedding_provider": "ollama",
    "embedding_model": "nomic-embed-text",
    "dimensions": 768,
    "api_base": "http://localhost:11434",
    "api_key": "...",
})
```

Only `api_base`, `api_key`, and `dimensions` are forwarded from the top level.
Other top-level keys (e.g., `backend`, `type`) are ignored.

When the nested format is present, it takes precedence over legacy keys.

### Embedding Mode

`CompletionMode.EMBEDDING` is always forced on the created provider, even if
the caller's config includes `"mode": "chat"`. This ensures the provider is
correctly configured for `embed()` calls.

### The vector width

`dimensions` says how wide the vectors should be. It is **honoured or
refused, never ignored** — by every provider, and by the per-call
`dimensions=` keyword as well as the config field.

```python
# Config-wide
provider = await create_embedding_provider(
    LLMConfig(provider="openai", model="text-embedding-3-large", dimensions=512)
)

# Or for one call, overriding the config
vectors = await provider.embed(texts, dimensions=256)
```

Which of the two happens depends on the model, and the answer is available
**before** you embed anything — which is what lets a fixed-width vector
column be created up front:

```python
from dataknobs_llm.llm.base import ModelCapability

if ModelCapability.EMBEDDING_DIMENSIONS in provider.get_capabilities():
    ...  # the width you ask for is the width you get
```

| The model | What a stated width does |
|---|---|
| accepts a width parameter (`text-embedding-3-*`, Titan V2) | forwarded to the API |
| has a fixed width (`text-embedding-ada-002`, Ollama, HuggingFace, Cohere Embed) | checked against what came back; a mismatch raises `ValueError` naming the model, the width asked for and the width returned |

Declaring the width a fixed-width model *does* produce is valid and silent —
the rule is that a stated width is never ignored, not that one may not be
stated. Stating nothing sends nothing and checks nothing, which matters for
`text-embedding-ada-002`: it rejects the parameter outright.

`EMBEDDING_DIMENSIONS` resolves from the bundled model tables and is
config-overridable through `model_profile_overrides`, so a model released
after the table was written can be declared without waiting for a release.

**Why this is not cosmetic.** The field was documented on `LLMConfig`, on
`create_embedding_provider` and on `AsyncLLMProvider.embed`, and was read by
one provider — Bedrock's Titan path. `EchoProvider` read a different key
(`options["embedding_dim"]`), and OpenAI, Anthropic, Ollama and HuggingFace
read neither. So a config asking `text-embedding-3-large` for 512 silently
received 3072: valid vectors, six times wider than requested, at six times
the storage and the price. Nothing raised at any layer; the first component
to object was a vector store rejecting the write, and that message names the
store rather than the misconfiguration.

### Config-lint validation

Because an embedder config is an `LLMConfig`, `dataknobs-llm` registers an
`"embedding"` resolver in `config_registries` (eager on import) that resolves
an `embedding` section to `LLMConfig` — the same resolver as the `"llm"`
binding. A consumer config that holds a nested `embedding` section
(currently `RAGKnowledgeBaseConfig` and `VectorMemoryConfig`) declares
`{"embedding": "embedding"}` in `_polymorphic_fields`, so
`config.validate()` dry-run-builds the embedder `LLMConfig` and surfaces an
unknown provider or bad field at config-parse time — without constructing a
provider. See the Structured Config guide in `dataknobs-common` for the
`validate()` / `_polymorphic_fields` mechanism.

### Examples

```python
from dataknobs_llm import create_embedding_provider

# Ollama with nomic-embed-text (default)
provider = await create_embedding_provider({
    "embedding": {
        "provider": "ollama",
        "model": "nomic-embed-text",
    },
})
embedding = await provider.embed("hello world")
await provider.close()

# OpenAI embeddings
provider = await create_embedding_provider({
    "embedding": {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "api_key": "sk-...",
        "dimensions": 1536,
    },
})

# Custom defaults (e.g., for testing)
provider = await create_embedding_provider(
    {},
    default_provider="echo",
    default_model="test-embed",
)
```

### Backward Compatibility

`create_embedding_provider()` is also available from `dataknobs_bots.providers`
for backward compatibility. The canonical import path is `dataknobs_llm`:

```python
# Preferred
from dataknobs_llm import create_embedding_provider

# Also works (backward compat)
from dataknobs_bots.providers import create_embedding_provider
```

## create_text_embedder()

Create an embedder — an embedding provider presented as the `TextEmbedder`
shape that `dataknobs-data`'s vector paths accept.

```python
from dataknobs_llm import create_text_embedder

embedder = await create_text_embedder(
    {"embedding": {"provider": "ollama", "model": "nomic-embed-text"}}
)

await db.bulk_embed_and_store(records, ["title", "body"], embedder=embedder)
```

### Signature

```python
async def create_text_embedder(
    config: LLMConfig | dict[str, Any],
    *,
    dimensions: int | None = None,
) -> LLMProviderEmbedder:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `LLMConfig \| dict` | — | Anything `create_embedding_provider()` accepts, including both dict formats above |
| `dimensions` | `int \| None` | `None` | The vector width, for a config that declares none |

There is deliberately **no new config type**. An embedder config *is* an
`LLMConfig`, which is what embedding providers were already configured by, so
this adds a runtime surface and not a configuration one.

### What it adds over `create_embedding_provider()`

The two things a bare provider cannot answer in the shape a *stored* vector
needs:

| | |
|---|---|
| `dimensions` | The vector width, settled. Taken from the `dimensions` argument, else the provider's configured `dimensions`, else observed on the first `embed()` — and a *declared* width is checked against that batch rather than trusted. The argument is the case that needs it: it names the width **this embedder** promises its callers, and the provider never sees it. A width in `config.dimensions` is reconciled one layer down (see [The vector width](#the-vector-width)) and arrives here already agreed. |
| `model_id` | `provider:model`, the **staleness key** written beside a stored vector. A sweep reads it back to decide whether a vector is still comparable, having never seen the embedder that produced it. |

`dimensions` raises if nothing has declared a width and nothing has been
embedded yet, rather than guessing.

### `LLMProviderEmbedder`

The adapter itself, for a provider you have already built:

```python
from dataknobs_llm import LLMProviderEmbedder

provider = await create_embedding_provider(config)
embedder = LLMProviderEmbedder(provider, model="text-embedding-3-small", dimensions=1536)
```

`model=` overrides the name reported in `model_id`. It does **not** change
which model the provider calls — it renames the vectors, so passing one that
does not match is how a staleness key comes to lie. Use it only where the
provider's own config understates the model actually in use.

There is no conversion in the adapter, and that absence is the point:
`AsyncLLMProvider.embed` already returns `list[list[float]]` for a list input,
which is exactly what `TextEmbedder.embed` returns. It satisfies the protocol
*structurally* rather than inheriting it — the one-directional edge is that
`data` cannot import `llm`, which is why the protocol lives there and the
implementation here. The protocol is imported under `TYPE_CHECKING` alone, so
the conformance is checked while nothing pulls `dataknobs_data.vector`, and
numpy behind it, into an `llm` import.

## Provider Backends

All three factory functions support all registered providers:

| Provider | Key | Package |
|----------|-----|---------|
| Ollama | `"ollama"` | Built-in |
| OpenAI | `"openai"` | Built-in |
| Anthropic | `"anthropic"` | Built-in |
| Amazon Bedrock | `"bedrock"` | Built-in (needs `[bedrock]` extra) |
| HuggingFace | `"huggingface"` | Built-in |
| Echo | `"echo"` | Built-in (testing) |

## Identifying a provider

A provider object answers two different questions about its identity, and
conflating them is a real defect source — a cost table keyed on the wrong one
silently prices every request at zero.

| Question | Accessor | Kind of answer |
|---|---|---|
| *What am I being billed by?* | `provider.provider_name` | Canonical **family** key — a closed set, lower-cased to match the key the registry resolved on |
| *What object is actually serving this call?* | `provider.impl_name` | Concrete **class** — an open set, including wrappers and consumer-registered providers |
| *What did the config author literally type?* | `provider.config.provider` | The verbatim configured string, untouched |

**Key lookup tables, metrics labels, and structured log fields on
`provider_name`.** `impl_name` is for diagnostics — log lines, error messages,
debugging output — and must never be a lookup key.

```python
from dataknobs_llm import LLMProviderFactory

provider = LLMProviderFactory(is_async=True).create(
    {"provider": "OpenAI", "model": "gpt-4o-mini"}
)

provider.provider_name      # 'openai'  — canonical, keyed on
provider.impl_name          # 'OpenAIProvider' — diagnostic only
provider.config.provider    # 'OpenAI'  — verbatim, as configured
```

`provider_name` is lower-cased deliberately. The provider registry resolves
classes case-insensitively but stores the config verbatim, so `provider: OpenAI`
and `provider: openai` select the same class while recording different strings.
Canonicalizing on read means a config author's shift key cannot split one
family's traffic across two rate-table rows or two metrics series.

The Python class name is **not** the identifier. It happens to resemble the
family key for the built-in providers (`OpenAIProvider` → `openai`) purely by
naming convention, and the resemblance breaks for wrappers:

```python
from dataknobs_llm.llm.providers.caching import (
    CachingEmbedProvider, MemoryEmbeddingCache,
)

wrapper = CachingEmbedProvider(provider, MemoryEmbeddingCache())

wrapper.provider_name   # 'openai' — still billed by OpenAI
wrapper.impl_name       # 'CachingEmbedProvider' — what handled the call
```

Both accessors are defined on `LLMProvider`, so every provider — async,
built-in, or consumer-registered — inherits them, and neither needs a
wrapper-side override.

`SyncProviderAdapter` is the exception, and it is the object
`create_llm_provider(..., is_async=False)` actually returns: it wraps an async
provider rather than subclassing `LLMProvider`, so it inherits nothing and
forwards both accessors explicitly. It reports the wrapped provider's family
and its own class, like any other wrapper.

### Declaring a family key

`provider_name` is assignable, for a provider whose family the config cannot
name — an OpenAI-compatible gateway configured as
`provider: openai-compatible` but billed as `acme`:

```python
class AcmeProvider(OpenAIProvider):
    def __init__(self, config):
        super().__init__(config)
        self.provider_name = "acme"
```

The assignment is canonicalized the same way a configured value is, so a
declared key obeys the same closed-set rule. Assign `None` to clear it and
fall back to the config.

## Model constraints (request-shape rules)

Some model families reject request parameters that others accept. The **Claude
5 family rejects `temperature`** with a hard 400, while the Claude 4.x family
(`claude-opus-4-8`, `claude-haiku-4-5-…`) still accepts it. A provider surfaces
these as a resolved `ModelConstraints` value (orthogonal to `ModelCapability`:
capabilities are the feature set; constraints are request-shape rules).

`ModelConstraints` carries:

| Field | Meaning |
|-------|---------|
| `rejected_params` | Generation/sampling params the family rejects — the provider **drops them before the call and logs a warning** (drop-and-warn, never silent). |
| `accepts_inline_system` | Whether the family accepts a `role="system"` message at a non-leading position (Anthropic hoists all system messages, so `False`). |
| `max_tokens_ceiling` | Upper bound on `max_tokens` for the model. When a request asks for more, the provider **clamps it down to the ceiling and logs a warning** (clamp-and-warn, never silent) — applied at a shared base choke point, so both the Anthropic and Bedrock (Claude) providers get it. Resolved from the live Models API on the native Anthropic endpoint (cached, TTL-refreshed) with a bundled fallback resource (the primary source on Bedrock); `None` (unknown model, or none overridden) leaves `max_tokens` untouched. |
| `param_remaps` | Wire-level `{canonical: wire}` parameter renames the family requires (e.g. the OpenAI reasoning families take `max_completion_tokens` in place of `max_tokens`). Applied **after** `adapt_config` at each provider's request-shaping choke point via the shared `LLMProvider._apply_param_remaps`, so a rename declared by a profile **or** a `LLMConfig.constraints` override is honored on any provider — not only the one that first needed it. Default empty (a no-op for families that need no rename). |

Both Claude providers — the native `AnthropicProvider` and the Bedrock provider
(Claude-on-Bedrock) — auto-detect the Claude 5 → `temperature`-rejection rule
from shared family knowledge, so a Claude-5-family config no longer 400s when it
carries `temperature:` — the param is dropped with a warning naming it and the
model:

```python
from dataknobs_llm import create_llm_provider

# temperature is dropped (with a logged warning) — no 400
provider = create_llm_provider({
    "provider": "anthropic",
    "model": "claude-sonnet-5",
    "temperature": 0.3,
})
```

Because the family table can go stale, the rule is **config-overridable** — a
consumer can declare a new rejected param or withdraw a stale one at runtime,
without waiting for a dataknobs release, via `LLMConfig.constraints` (a loose
dict resolved at runtime, the same mechanism as `capabilities`):

```python
# Add a rejected param the built-in table doesn't know about yet:
create_llm_provider({
    "provider": "anthropic",
    "model": "claude-6-future",
    "constraints": {"rejected_params": ["temperature", "top_p"]},
})

# Withdraw a stale rule (send temperature to a Claude 5 model anyway):
create_llm_provider({
    "provider": "anthropic",
    "model": "claude-sonnet-5",
    "temperature": 0.3,
    "constraints": {"rejected_params": []},
})
```

The *merge* semantics differ from `capabilities`, deliberately: a `capabilities`
override **replaces** the detected list wholesale, whereas a `constraints`
override is **overlaid per field** — an absent override key keeps the
auto-detected value. So `{"accepts_inline_system": false}` leaves the detected
`rejected_params` in force rather than resetting the whole structure. (Within
the override, `rejected_params` itself is replaced by the list you supply — pass
`[]` to withdraw a stale rule, as above.)

Constraints resolve from the **per-call runtime config**, so a call that
overrides the model to a different family (`complete(..., config_overrides={"model": ...})`)
gets that family's rules — the drop reflects the model actually being sent, not
just the configured default.

The same surface carries `max_tokens_ceiling`. When a request's `max_tokens`
exceeds the model's ceiling, the provider **clamps it down to the ceiling and
logs a warning** — clamping *down* is always a valid request (asking for fewer
output tokens never 400s), so this pre-empts the output-truncation / 400 class
at source rather than recovering from it. The clamp (and the rejected-param
drop) apply in canonical config space at a shared base choke point
(`LLMProvider._apply_request_constraints`), so the **same** behavior serves both
Claude providers — the native **Anthropic** Messages API and **Bedrock**
Converse (Claude-on-Bedrock) — from one implementation. Non-Claude Bedrock
models (Llama, Mistral, Nova, Titan, Cohere, AI21) resolve their ceilings from
the bundled `bedrock_models.yaml` resource (see the Bedrock binding above); an
unlisted model resolves a permissive `None` ceiling.

The ceiling is **resolved dynamically**, in this precedence per model:

1. **Config override** — `LLMConfig.constraints={"max_tokens_ceiling": N}` always
   wins over any dynamically-resolved value.
2. **Live Models API** *(native Anthropic endpoint only)* — the Anthropic Models
   API reports each model's `max_tokens`. The provider caches it per process and
   refreshes it on a TTL (at most one `models.list()` per TTL per event loop,
   never per request; each poll independently timeout-bounded so a hung API never
   stalls the request path), so a ceiling that changes between releases is picked
   up without a dataknobs release. Bedrock has no Models API, so it resolves
   directly against the bundled resource below.
3. **Bundled fallback resource** — a maintained data file shipped with the
   package (`llm/providers/data/anthropic_model_limits.yaml`), used when the
   dynamic path has produced no value (no API key, offline, an API blip) and as
   the primary source on Bedrock. The dynamic cache and the resource share one
   family-matching rule (exact id, then longest family-substring in either
   direction), so a bare model alias resolves a dated key and vice versa. A
   known-good dynamic value is never degraded back to the resource on a transient
   failure.
4. **`None`** → permissive, `max_tokens` passes through untouched — identical to
   the pre-clamp default (which sends `1024`, well under any real ceiling, so the
   overwhelming majority of requests are byte-identical).

`initialize()` performs no network I/O; the ceiling is refreshed lazily at the
first request boundary (before the clamp), so the first completion already
clamps against fresh data.

```python
# Pin the ceiling explicitly — an over-ceiling max_tokens is clamped down (with
# a logged warning) instead of truncating or 400-ing. The override always wins
# over any dynamically-resolved value:
create_llm_provider({
    "provider": "anthropic",
    "model": "claude-sonnet-5",
    "max_tokens": 100_000,
    "constraints": {"max_tokens_ceiling": 8192},  # request clamped to 8192
})
```

Two provider `options` tune the dynamic path (both optional):

| Option | Default | Effect |
|--------|---------|--------|
| `model_limits_ttl` | `3600` | Seconds between Models-API refreshes — the freshness-vs-traffic knob. Near-zero re-fetches each call; large minimizes API traffic. |
| `model_limits_refresh_timeout` | `10` | Seconds a single Models-API refresh poll may run before it is abandoned (falling back to the cached/resource value). Bounds a hung control plane independently of the request `timeout`. |
| `model_limits_dynamic` | `true` | Set `false` to disable Models-API calls entirely (resource-only). |

A consumer that prefers to drive freshness on its own schedule can call
`await provider.refresh_model_limits()` to force an immediate refresh.

The bundled fallback resource is kept current with a maintainer tool that
reconciles it against the live API — `bin/update-model-limits.sh --check` reports
drift (a key-gated CI signal) and `--update` rewrites the file from live values.
Both are a clean no-op when `ANTHROPIC_API_KEY` is unset.

### Unified model-metadata substrate (`ModelProfile`)

Capabilities, request-shape rules, token ceilings, and pricing are all
*model-keyed facts* that go stale on each vendor release. Rather than each
provider hand-maintaining scattered literals for each, they resolve through one
substrate (`dataknobs_llm.llm.model_profile`): a single `ModelProfile` record
holding every facet, resolved by a `LayeredModelProfileResolver` that merges an
ordered list of sources **facet-by-facet, highest precedence first**.

| `ModelProfile` facet | Feeds |
|----------------------|-------|
| `context_window` | `ModelConstraints.max_input_tokens` |
| `max_output_tokens` | `ModelConstraints.max_tokens_ceiling` |
| `capabilities` | `get_capabilities()` |
| `rejected_params` | `ModelConstraints.rejected_params` (family rule) |
| `param_remaps`, `pricing`, `available`, `aliases` | (per-provider bindings) |

The merge is **override, not union**: for each facet the first source with a
non-`None` value wins, and no lower source can displace it. `None` means
"unknown"; a *present* value — including an empty `frozenset()` — means
"authoritatively known," so a config that pins `capabilities=frozenset()`
("this model has none") replaces a lower layer's guess. `AnthropicProvider`
composes its resolver as **config override → live Models-API cache → bundled
resource → heuristic**, so the ceiling facets resolve live-else-resource while
capabilities and the `temperature` rule come from the heuristic — all overridable
per facet.

The highest-precedence layer is `LLMConfig.model_profile_overrides`, a loose
mapping that lets a consumer supply or correct *any* facet without a dataknobs
release — either a flat facet mapping (applies to the configured model) or a
`{model_id: {facets}}` per-model mapping:

```python
create_llm_provider({
    "provider": "anthropic",
    "model": "claude-sonnet-5",
    # correct one facet; everything else still resolves normally
    "model_profile_overrides": {"max_output_tokens": 8192},
})
```

Sources implement the `ModelMetadataSource` protocol (a synchronous, I/O-free
`resolve(model) -> ModelProfile`; a live-backed source refreshes its cache
out-of-band). An in-house gateway or proxy registers its own source via the
consumer-extensible `model_metadata_sources` registry — no dataknobs release
required. This is the substrate every provider binds to as it is migrated onto
it.

For a facet a vendor serves **live**, the built-in `LiveApiSource`
(`from dataknobs_llm.llm import LiveApiSource`, alongside its sibling sources) is
the reusable live layer: construct it with an async `list_models()` and a
`(api_object) -> ModelProfile` extractor, and it carries the refresh machinery —
TTL-gated polling (a fresh cache is a no-op; at most one poll per TTL per event
loop), per-loop-locked dedup (concurrent cold-cache callers coalesce into one
poll), a bounded poll (`refresh_timeout`), and source-aware non-degradation (a
transient refresh failure leaves a known-good live value intact rather than
dropping to the bundled fallback). `resolve` reads the cache synchronously; the
provider drives `refresh_if_stale()` / `force_refresh()` from its async request
boundary. `AnthropicProvider` composes one bound to its Models-API listing to
source the two token ceilings; each provider owns its own instance (its own
cache). A vendor whose id space collides under the default substring
family-matcher (Ollama's `name:tag` ids) injects its own `match=(model, keys) ->
key | None` rule — the default is `match_family_key`, byte-identical for every
other adopter. The same `match=` seam is available on `ConfigOverrideSource`
(the config-override layer), so a provider whose **per-repo override map** keys
collide under substring matching injects an exact matcher there too —
HuggingFace does, for its prefix-sharing repo ids.

#### OpenAI binding

`OpenAIProvider` binds to the substrate with a **maintained-fallback** resolver —
**config override → bundled resource → heuristic**. There is no `LiveApiSource`:
OpenAI's Models API serves only model *ids* (no ceilings / capabilities /
pricing), so the bundled `openai_models.yaml` resource is the primary declarative
source, a corrected family heuristic backs unlisted models, and
`model_profile_overrides` wins per facet. The binding turns on capabilities for
current families the old substring lists missed (`gpt-5` / o-series tools, JSON,
vision), the `max_tokens` clamp + input budget (previously dead for OpenAI), and
two request-shape rules for the reasoning families: `rejected_params`
(`temperature` / `top_p` are dropped) and `param_remaps` — a **wire-level** rename
declared as data (`{"max_tokens": "max_completion_tokens"}`) applied after
`adapt_config` by the shared `LLMProvider._apply_param_remaps`. An unknown model
resolves an all-permissive profile, so it is shaped exactly as before.
`validate_model` still lists the Models API by default, but now honors a
`model_profile_overrides.available` pin first (via the shared
`ProfileDetectionMixin.validate_model`), so a consumer on a private gateway can
skip the round-trip — consistent with HuggingFace / Ollama / Bedrock.

```python
# max_tokens is renamed to max_completion_tokens and temperature is dropped —
# no 400 — for an o-series request; a gpt-4o request is clamped to its ceiling.
create_llm_provider({"provider": "openai", "model": "o1"})
```

#### Bedrock binding

Bedrock serves **two model populations** behind one provider, and the binding
sources a different slice of each. For **non-Claude** families (`amazon.nova-*`,
`amazon.titan-*`, `meta.llama*`, `mistral.*`, `cohere.*`, `ai21.*`) the bundled
`bedrock_models.yaml` resource carries the **full** profile — capabilities,
output/context ceilings, and pricing. For **Claude-on-Bedrock**
(`anthropic.claude-...`) it carries **only** the Bedrock-owned facets (`pricing`,
`available`); the capabilities, output ceiling, context window, and Claude-5
`temperature` rejection come from the **shared Claude sources** that the native
`AnthropicProvider` also composes — a Claude model's family facts are a property
of the model, not the endpoint, so they are never re-copied (no drift). The
resolver is **config override → bedrock resource → shared Claude ceiling → shared
Claude (Claude-only) capabilities → bedrock heuristic**, first-non-`None` per
facet.

The binding fixes vision detection for the multimodal non-Claude families the old
substring list missed (Nova lite/pro/premier, Llama-3.2 vision, Pixtral),
populates `max_input_tokens` for Bedrock (the input budget was previously dead),
wires `cost_usd` off the resolved per-Mtok `ModelPricing` on **both** the buffered
and streaming paths, and turns `validate_model` from a hardcoded prefix whitelist
into a data-sourced `available` read. A cross-region inference-profile id
(`us.`/`eu.`/`apac.`/`us-gov.`) resolves the same family as its base id.

**Opt-in live availability.** By default `validate_model` makes **no**
control-plane call — `bedrock:ListFoundationModels` is a permission distinct from
`bedrock:InvokeModel`, so a live default would break least-privilege
inference-only roles. Set `options["model_availability_live"] = true` to validate
against the account's live `ListFoundationModels` catalog instead (a model absent
from the account resolves `False`); it reuses the substrate's TTL / per-loop-lock
refresh. Capabilities and ceilings stay maintained-resource regardless — the live
catalog serves availability + modalities, not ceilings.

```python
# Default: data-sourced availability, no control-plane call.
create_llm_provider({"provider": "bedrock", "model": "amazon.nova-pro-v1:0"})

# Opt in to live catalog validation (needs bedrock:ListFoundationModels):
create_llm_provider({
    "provider": "bedrock",
    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "options": {"model_availability_live": True},
})
```

#### Ollama binding

Ollama is **local and live-first** — the only binding whose live source is the
*primary* layer. The Ollama server authoritatively reports each installed model's
capabilities and context window via `POST /api/show` (a `capabilities` array like
`["completion","tools","vision","embedding"]` plus
`model_info.<arch>.context_length`), so the provider composes a `LiveApiSource`
that walks `GET /api/tags` (the installed set) and enriches each model with
`/api/show`. The resolver is **config override → live `/api/show` cache →
name-based heuristic** — there is **no** bundled resource (Ollama's model space is
open-ended and user-pulled) and **no** pricing / output-ceiling layer (local/free,
and `num_predict: -1` means unlimited output — no ceiling to clamp).

The binding replaces the hardcoded capability-substring lists that rotted each
release: modern families the old lists missed (`llama4`, `gpt-oss`, `qwen3`, …)
are now tool/vision-detected from the server's own report, `max_input_tokens` is
populated from the reported context window (previously dead for Ollama), and
`validate_model` reads the resolved `available` facet (installed → `True`;
not-installed / unreachable → `False`), force-refreshing the live cache first so
a model pulled since the last request is seen immediately (an authoritative
liveness check, not a value that can lag by up to the metadata TTL). A dedicated
embedding model resolves an `EMBEDDINGS`-only set. The name-based heuristic is
the graceful-degradation fallback for older servers that predate the
`capabilities` field — or any server reporting no usable capability array (an
empty or all-unrecognized report degrades to the heuristic rather than resolving
the model to zero capabilities).

Because Ollama ids are `name:tag` (e.g. `llama3.1:8b`), the live source is
constructed with a `name:tag`-aware `match=` matcher — the default substring
matcher would false-resolve `nomic-embed-text` to `nomic-embed-text-v2-moe:latest`
(a substring collision). The live cache is tunable via `options`
(`model_metadata_live` to disable, `model_metadata_ttl`,
`model_metadata_refresh_timeout`). Ollama sources no pricing of its own, but a
consumer can still model private GPU cost through
`model_profile_overrides.pricing`, which lights up `get_pricing` / `estimate_cost`.

```python
# Capabilities + context window come live from the local server; a bare alias
# (llama3.1) resolves an installed tagged model (llama3.1:8b).
create_llm_provider({"provider": "ollama", "model": "llama3.1"})

# Model private GPU cost (Ollama serves no pricing of its own):
create_llm_provider({
    "provider": "ollama",
    "model": "llama3.1:8b",
    "model_profile_overrides": {"pricing": {"input_per_mtok": 0.0, "output_per_mtok": 0.0}},
})
```

#### HuggingFace binding

HuggingFace is **heuristic-primary + override-rich** — the leanest binding, and
the last provider migrated off the inline capability substring lists. Its model
space is millions of community repos with no vendor catalog, and the serverless
Inference API serves no offered-set / ceiling / pricing endpoint, so there is
**no live source and no bundled resource**: the resolver is just **config
override → repo-name capability heuristic**. Every non-capability facet
(context window, rejected params, param remaps, pricing, availability) is `None`
from the heuristic and lit up **only** by `model_profile_overrides` — exactly
right for a provider whose "catalog" is whatever repo the consumer points at.

The corrected heuristic classifies from the repo name and emits the complete
capability set: `TEXT_GENERATION` always; `EMBEDDINGS` for the dominant embedding
families (`sentence-transformers/*`, `feature-extraction`, and the `minilm` /
`bge` / `gte` / `e5` / `instructor` family markers), **excluding** cross-encoder
rerankers (any embed-marker match is dropped when the repo also carries a
`reranker` token — a reranker is not an embedding model); `CHAT` for a `chat` /
`instruct` / `conversational` **substring**, so fused real-world names such as
`chatglm3` and `openchat` keep resolving `CHAT`. The embedding-family name markers
(`minilm` / `bge` / `gte` / `e5` / `instructor`) are matched at token boundaries
(so a short marker like `e5` does not fire inside an unrelated `phase5` run),
while the longer descriptive markers (`sentence-transformers/`,
`feature-extraction`) match as substrings. `EMBEDDINGS` and `CHAT` are **structurally
disjoint** — an embedding repo never also resolves `CHAT` (the Inference API
serves a repo as one task), a property guaranteed by the logic (embed is resolved
first and suppresses chat), not merely by the tested names. That embed-first
ordering also neutralizes the `instruct` ⇄ `instructor` collision: `instructor`
is an embed token, resolved before the chat substring is ever checked. It deliberately asserts **no** `STREAMING` (HF's `stream_complete` is a
simulated single yield, not real token streaming) and **no** `FUNCTION_CALLING`
(the Inference API rejects tools); `VISION` / `CODE` / `JSON_MODE` are declared per
repo via `model_profile_overrides.capabilities`.

The binding also replaces the hardcoded `max_new_tokens=100` output default with
a named constant routed through the shared request-shaping choke point (a
consumer's `constraints.rejected_params` / `param_remaps` are now honored — a
byte-identical no-op otherwise), and `validate_model` keeps its authoritative
`GET {base}/{model}` liveness probe but honors a `model_profile_overrides.available`
pin (a private-gateway / TGI consumer that knows its model is live and wants to
skip the probe). That pin-honoring is the shared `ProfileDetectionMixin.validate_model`
behavior: a substrate-bound provider whose profile has no source populating
`available` (HuggingFace, OpenAI) inherits it and overrides only the network probe
(`_probe_model_available`); a provider whose profile resolves `available` from a
live / resource source (Ollama, Bedrock) reads the facet directly. HuggingFace
sources no pricing of its own, but a consumer can model private-endpoint cost
through `model_profile_overrides.pricing`, which lights up `get_pricing` /
`estimate_cost`.

Because HuggingFace repo ids are exact strings that share prefixes
(`meta-llama/Llama-3.1-8B` is a substring of `meta-llama/Llama-3.1-8B-Instruct`),
the config-override source is constructed with the same injectable `match=` seam
the Ollama live source uses — here an **exact repo-id matcher** — so a per-repo
override map does not resolve a base repo to a prefix-sharing variant's override.
The `match=` argument on `ConfigOverrideSource` defaults to `match_family_key`
(byte-identical for every other provider); only HuggingFace opts into exact
matching.

```python
# Repo-name heuristic: an instruct repo resolves {TEXT_GENERATION, CHAT};
# a sentence-transformers repo resolves {TEXT_GENERATION, EMBEDDINGS}.
create_llm_provider({"provider": "huggingface", "model": "mistralai/Mistral-7B-Instruct-v0.2"})

# Declare per-repo facts the heuristic cannot know (context window, vision,
# private-endpoint pricing, a live-availability short-circuit):
create_llm_provider({
    "provider": "huggingface",
    "model": "llava-hf/llava-1.5-7b-hf",
    "model_profile_overrides": {
        "capabilities": ["text_generation", "chat", "vision"],
        "context_window": 4096,
        "available": True,
    },
})
```

> **Deferred — the live Hub source.** HuggingFace's authoritative live signal is
> a **per-model** Hub lookup (`GET huggingface.co/api/models/{id}` →
> `pipeline_tag` / `tags` / `config.max_position_embeddings`), a fundamentally
> different shape from the walker-based `LiveApiSource` (per-model on-demand
> fetch, not list-all-and-cache) on a second host. It is a captured follow-up
> awaiting its own design pass; today per-repo facts come from
> `model_profile_overrides`.

#### Pricing (`get_pricing` / `estimate_cost`)

Pricing is unified on `ModelPricing` (per-million-token) and reachable through the
provider without touching its resolver. `provider.get_pricing(model=None)` reads
the resolved profile's `pricing` facet (the **facts** accessor, symmetric with
`get_constraints`); `provider.estimate_cost(response, model=None)` is the one-call
**convenience** that resolves pricing and computes the cost via the
provider-agnostic `CostCalculator`. For costing a stored/replayed response
offline, call `CostCalculator.calculate_cost(response, pricing=...)` directly with
a `ModelPricing`; passing no `pricing` falls back to a small built-in table.

```python
llm = create_llm_provider({"provider": "openai", "model": "gpt-4o"})
cost = llm.estimate_cost(response)          # resolves gpt-4o pricing, then computes
price = llm.get_pricing("gpt-4o-mini")      # ModelPricing (facts only)
```

### Vendor-error translation (all providers)

Every provider translates raw vendor transport errors into
`dataknobs_common.exceptions` types, so a consumer catches by a dataknobs type
without importing a vendor SDK (the original error is preserved on `__cause__`):

| Vendor error | dataknobs exception |
|--------------|---------------------|
| 400 — context-window overflow | `ContextLengthExceededError` (a `ValidationError` subclass) |
| 400 (other bad request) | `ValidationError` |
| 429 (rate limit) | `RateLimitError` (with `retry_after` when the vendor exposes it) |
| 401 / 403 (auth) | `OperationError` |
| other status / connection / timeout | `OperationError` |

A **context-window overflow** — the request's input exceeded the model's
maximum context length — is a 400, so it is a `ValidationError`; the translator
raises the narrower `ContextLengthExceededError` (`from dataknobs_llm.exceptions
import ContextLengthExceededError`) for it. Because that type *is a*
`ValidationError`, an existing `except ValidationError` keeps matching — catch
the narrower type only when you want to react to overflow specifically (compact
history and retry, switch to a larger-context model, or surface a distinct
message). Detection is a machine `code` (OpenAI) or a conservative marker in the
vendor's own text (all vendors), and stays deliberately narrow — an unrelated 400 (a
rejected sampling parameter, a malformed request) remains a plain
`ValidationError`.

This is uniform across Anthropic, OpenAI, Ollama, HuggingFace, and Bedrock: the
status→type policy lives once on `LLMProvider._dataknobs_error_for_status`, and
each provider adds only a small SDK-specific extractor — the Anthropic / OpenAI
`APIError` subtree, aiohttp's `ClientResponseError` for Ollama / HuggingFace, and
botocore's nested `ClientError` status for Bedrock (whose throttling *codes*
`ThrottlingException` / `TooManyRequestsException` also map to `429`). It covers
every request entry point — `complete`, `stream_complete`, `embed`, and the
deprecated `function_call`. For the streaming path a vendor error is translated
whether it surfaces at stream *creation* or partway through *iteration* (both
run through the shared `_call_api` / `_iter_translated` choke points), so a
mid-stream rate limit or dropped connection is a dataknobs exception too. A
non-vendor exception (a bug in caller code) propagates unchanged rather than
being masked as an API error. When a `429` carries a `Retry-After` header,
`retry_after` is parsed from either form the RFC permits — a number of seconds
or an HTTP-date (converted to seconds-from-now).

Domain-specific errors are raised *ahead of* the translator and never flattened:
Ollama's / HuggingFace's `ToolsNotSupportedError` (a model that cannot do tool
calling) stays a `ToolsNotSupportedError`, not a generic `ValidationError`.

**Migrating from raw vendor `except` blocks.** If you previously caught a raw
vendor type around a provider call (`except openai.RateLimitError`, `except
aiohttp.ClientResponseError`, `except botocore.exceptions.ClientError`), switch
to the dataknobs type (`except RateLimitError` / `except OperationError` from
`dataknobs_common.exceptions`); the raw error is still reachable on `__cause__`.

**What the message says, and what it doesn't.** A translated error's message is
built from the provider family and the status — `"openai API error (HTTP 400)"`,
or `"ollama API error"` when the failure carried no status; a context-window
overflow adds `": request exceeds the model's context window"`, the one 400
worth distinguishing in the text because the caller can act on it. It
deliberately does
**not** include the vendor's own rendering, because two of the types above are
rendered *with their message shown* at an HTTP boundary (`ValidationError` at
422 and `RateLimitError` at 429 under the `dataknobs-bots` API layer's default
policy), and a vendor rendering is not ours to disclose: `aiohttp` renders the
endpoint URL, and the OpenAI and Anthropic SDKs relay the response body
verbatim. The rendering is not lost — it is on `__cause__`, so
`raise ... from exc` chaining puts it in any traceback, and the bots API layer
appends it to the log line it writes for the errors it maps to a status — which
is the path a translated error takes. (The `HTTPException` and catch-all
handlers write no `cause=` field; they are not what these errors reach.)

The family key comes from `provider.provider_name`, so a gateway that declares
its own key reports that key here too. If you write your own provider, note that
`_dataknobs_error_for_status(status, detail, ...)` takes the vendor rendering as
*classification* material only — it decides context-window overflow from it and
then discards it. You cannot set the message, which is the point: a provider
this package has never seen inherits the same guarantee.

#### Anthropic 400-retry safety net

As a safety net for a **model family the constraint table doesn't know yet**, the
`AnthropicProvider` recovers an "unsupported sampling parameter" 400 once: the
offending param is dropped, the request retried, a warning logged, and the
discovery memoized for the process so subsequent requests to that model drop it
up front (≤1 wasted round-trip per model). Declaring the param in `constraints`
pre-empts even that first round-trip. (`function_call` still falls back to
prompt-based function calling on a `400`, the "older model lacks the native
tools API" signal, but a `429` / auth error propagates as its translated
dataknobs exception instead of triggering a second API call.)

## Response truncation signal

When a provider cuts generation off at the token budget, the response is
**incomplete** — and the most dangerous case is a truncated *tool-call* turn,
whose partial arguments look well-formed but are invalid (e.g. a required field
missing), so the model's request fails a downstream validator with no hint that
truncation was the cause. Every provider surfaces this with a single boolean
that does not depend on knowing each provider's stop-reason vocabulary:

```python
response = await llm.complete("...", tools=my_tools)
if response.truncated:
    # Do NOT feed response.tool_calls to the tool — the arguments are partial.
    # Raise max_tokens (or shorten the request) and retry.
    ...
```

`LLMResponse.truncated` (and `LLMStreamResponse.truncated` on the final chunk)
is populated consistently:

| Provider | Truncation condition |
|----------|----------------------|
| Anthropic | `stop_reason == "max_tokens"` |
| OpenAI | `finish_reason == "length"` |
| Ollama | `done_reason == "length"` |
| Bedrock | `stopReason == "max_tokens"` |

A truncated tool-call turn is logged at `warning`; a truncated plain-text turn
at `info`. The HuggingFace inference path returns no stop-reason signal, so
`truncated` is always `False` there.

The Claude-family providers (`AnthropicProvider` and `BedrockProvider`) also
normalize `finish_reason` onto the canonical vocabulary the `LLMResponse`
docstring advertises (`max_tokens` → `length`, `tool_use` → `tool_calls`,
`end_turn`/`stop_sequence` → `stop`); the raw provider value is preserved on
`metadata["raw_finish_reason"]`. They share Claude's stop-reason vocabulary
verbatim (Bedrock runs Claude), so both route through a single shared helper
(`normalize_claude_stop_reason`) rather than each maintaining its own table.
OpenAI and Ollama already report the canonical vocabulary directly, so
`finish_reason` reads identically across every provider.

> For how truncation handling, model constraints, and history bounds fit
> together when building a long-running tool bot, see the LLM Best Practices
> guide's [Productionizing a Tool-Using Bot](best-practices.md#productionizing-a-tool-using-bot)
> checklist.

## Mid-conversation system-message policy (Anthropic)

Anthropic's Messages API has no inline `system` role — a system prompt is a
top-level `system` parameter, not a message in the array. A **leading** system
message is therefore always hoisted into that parameter (correct and required).
A **mid-conversation** system message — a positional, in-context notice
appended *after* the dialog has started ("the loop timed out; use the results
already available") — has no natural home, and historically was silently
hoisted into the global system prompt too, turning an event-at-a-point into a
standing instruction with no warning.

`AnthropicProvider` makes this a configurable policy via
`options["system_message_policy"]`:

```python
config = LLMConfig(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    options={"system_message_policy": "inline"},  # the default
)
```

| Policy | Mid-conversation `role="system"` behavior |
|--------|-------------------------------------------|
| `inline` (**default**) | Convert to a `user` message **at its position**, preserving the in-context meaning. Content blocks are consolidated so the request stays valid: no consecutive same-role messages; every `tool_result` stays adjacent/paired to its `tool_use`; and `tool_result` blocks are kept **first** in a user turn (with the inlined notice text after them), as Anthropic's Messages API requires. |
| `hoist` | Merge into the top-level `system` param (legacy behavior; positionally lossy but byte-for-byte back-compatible). |
| `warn` | Log a warning naming the message, then hoist — makes the lossy case visible without changing structure. |
| `reject` | Raise `ValidationError` — treat a mid-conversation system message as a configuration error. |

An unrecognized policy raises `ValidationError` at provider construction
(fail-closed). Whether a model family accepts an inline system message at all is
the S1 `ModelConstraints.accepts_inline_system` datum (`False` for Anthropic);
a consumer that declares `constraints={"accepts_inline_system": True}` opts the
family out of the policy entirely — a mid-conversation system message is then
left in place.

> The default changed from `hoist` to `inline`: `inline` preserves the notice's
> positional meaning and the consolidation keeps the adapted request
> structurally valid, so the more-correct behavior carries no alternation risk.
> Set `system_message_policy: "hoist"` to restore the exact legacy shape.

The `tool_use` ↔ `tool_result` pairing invariant this consolidation relies on
is a shared, provider-agnostic function
(`dataknobs_llm.llm.message_sequence.pair_orphan_tool_calls`) so the same rule
is enforced in one place across the reasoning strategies and the adapters.

## Amazon Bedrock

Amazon Bedrock is registered as the `"bedrock"` provider. A single
`BedrockProvider` serves **both** chat/completion (via the unified Converse
API) and embeddings (Titan / Cohere via `invoke_model`).

**Authentication is via the AWS credential chain — there is no API key.**
Credentials resolve from the environment, the `~/.aws` shared config, or an
EC2/ECS instance or task IAM role. Region, endpoint, explicit credentials, and
Bedrock guardrail settings are supplied through `LLMConfig.options`.

Install the async Bedrock transport (`aioboto3`, pulled via the shared
`dataknobs-common[aws]` session factory):

```bash
pip install 'dataknobs-llm[bedrock]'
```

### Chat / completion

```python
from dataknobs_llm import create_llm_provider

provider = create_llm_provider({
    "provider": "bedrock",
    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "temperature": 0.7,
    "max_tokens": 1024,
    "options": {"region_name": "us-west-2"},  # credentials via the IAM chain
})
await provider.initialize()
response = await provider.complete("Explain quantum computing")
```

The model id is a Bedrock foundation-model id
(`anthropic.claude-3-5-sonnet-20240620-v1:0`) or a cross-region
inference-profile id (`us.anthropic.claude-3-5-sonnet-20240620-v1:0`).
Streaming (`stream_complete`) and tool use (`complete(tools=...)`) work as with
the other providers.

### `options` keys

| Key | Purpose |
|-----|---------|
| `region_name` (or `region`) | AWS region for the client |
| `endpoint_url` | Custom endpoint (PrivateLink / VPC endpoint). Bedrock's endpoint knob — `LLMConfig.api_base` is not consulted |
| `aws_access_key_id` / `aws_secret_access_key` / `aws_session_token` | Explicit credentials (omit to use the credential chain). A partial pair fails closed at construction |
| `normalize` | Titan embeddings only — L2-normalize the vector (default `True`) |
| `input_type` | Cohere embeddings only — `"search_document"` (default) or `"search_query"` at query time |
| `embed_max_concurrency` | Bound on Titan's per-text `invoke_model` fan-out (default: `max_pool_connections`, i.e. `10`) |
| `stream_read_timeout` | Per-socket-read (inter-chunk) timeout for `stream_complete`, in seconds (default: boto's `60`s). See the timeout note below |
| `guardrail_identifier` + `guardrail_version` | Applied to Converse requests when both are set (optional `guardrail_trace`) |

The `complete()` / `function_call()` socket read timeout is `LLMConfig.timeout`
(default `60`s); retry and connection-pool tuning follow the shared
`AwsSessionConfig` defaults. **Streaming is different:** botocore's
`read_timeout` is a per-read (inter-chunk) timeout, and there is no
total-stream-duration knob, so `LLMConfig.timeout` is *not* applied to
`stream_complete` — otherwise a long inter-token pause would kill the stream.
Streaming uses `stream_read_timeout` instead (default: boto's `60`s); raise it
for slow-thinking models.

### Embeddings

```python
from dataknobs_llm import create_embedding_provider

provider = await create_embedding_provider({
    "embedding": {
        "provider": "bedrock",
        "model": "amazon.titan-embed-text-v2:0",
        "dimensions": 1024,
        "options": {"region_name": "us-west-2"},
    },
})
vector = await provider.embed("hello world")
```

Two embedding families are supported:

| Family | Model ids | Notes |
|--------|-----------|-------|
| Amazon Titan | `amazon.titan-embed-text-v2:0` | `dimensions` selects 256 / 512 / 1024 (default 1024); embeds one text per call, bounded by `embed_max_concurrency`; `normalize` via options |
| Cohere | `cohere.embed-english-v3`, `cohere.embed-multilingual-v3` | embeds the whole list in one call; `input_type` via options (`search_query` at query time) |

An unrecognized embedding-model id raises `ValueError` naming the two supported
families.

## Testing

Use `EchoProvider` (via `"echo"` provider key) for tests:

```python
provider = await create_embedding_provider({
    "embedding": {"provider": "echo", "model": "test"},
})
embedding = await provider.embed("test input")
assert len(embedding) > 0
await provider.close()
```

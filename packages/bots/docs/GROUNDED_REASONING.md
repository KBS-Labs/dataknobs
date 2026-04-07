# Grounded Reasoning Strategy

The grounded reasoning strategy guarantees that every substantive turn retrieves from configured data sources, eliminating the unreliability of LLM-decided retrieval. Unlike ReAct (where the model may skip KB search), grounded retrieval is structural — a pipeline step that always executes.

## When to Use Grounded vs. Other Strategies

| Strategy | Use When |
|----------|----------|
| **Grounded** | Responses MUST be grounded in KB content with provenance. Research analysts, compliance, documentation Q&A, "cite your sources" requirements. |
| **ReAct** | Flexible multi-step reasoning with diverse tools (calculator, API calls, code execution). KB search is one of many optional tools. |
| **Simple** | Direct LLM conversation without tools or retrieval. |
| **Wizard** | Structured data collection with FSM-driven stage progression. |

## Quick Start

### Minimal Configuration

```yaml
llm:
  provider: ollama
  model: llama3.2

knowledge_base:
  path: docs/
  chunker:
    strategy: markdown
    max_chunk_size: 800
  embeddings:
    provider: ollama
    model: mxbai-embed-large

reasoning:
  strategy: grounded
  intent:
    mode: extract
    num_queries: 3
    domain_context: "OAuth 2.0 authorization framework"
  retrieval:
    top_k: 5
    score_threshold: 0.3
  synthesis:
    require_citations: true
  store_provenance: true
```

### Programmatic Usage

```python
from dataknobs_bots import DynaBot

bot = await DynaBot.from_config(config)

# Chat — retrieval always happens
response = await bot.chat("What are the OAuth grant types?", context)

# Streaming — intent + retrieval run first, then synthesis streams
async for chunk in bot.stream_chat("Compare auth code vs implicit", context):
    print(chunk.content, end="")

# Access provenance
manager = bot.get_conversation_manager(context.conversation_id)
provenance = manager.metadata.get("retrieval_provenance")
print(f"Queries: {provenance['intent']['text_queries']}")
print(f"Results: {provenance['total_results']} → {provenance['deduplicated_to']}")
```

## Pipeline

Every turn executes this pipeline:

```
User message + conversation history
  |
  +-- 1. Intent Resolution
  |     Determine WHAT to search for (one of three modes)
  |
  +-- 2. Multi-Source Retrieval
  |     Execute intent against all configured sources
  |     Round-robin interleave + deduplicate
  |     Merge adjacent chunks + format for LLM
  |
  +-- 3. Synthesis
  |     Generate response from retrieved context (LLM or template)
  |
  +-- 4. Provenance Record
        Log intent, results, scores, timing
```

## Intent Resolution

Intent resolution determines what to search for. Three modes are available, selectable via `intent.mode`.

### Extract Mode (Default)

An LLM generates search queries from the user message. This is the most flexible mode.

```yaml
reasoning:
  strategy: grounded
  intent:
    mode: extract
    num_queries: 3              # Generate up to 3 queries
    domain_context: "OAuth 2.0" # Domain hint for the query gen prompt
    use_conversation_context: true  # Include recent history
```

The LLM receives a structured prompt emphasizing "underlying intent, not literal text", "related topics", and "key concepts". By default, this uses the `QueryTransformer` from `dataknobs_bots.knowledge.query` with text-based parsing.

**Optional: Structured intent extraction via SchemaExtractor**

When `extraction_config` is provided, extract mode uses a `SchemaExtractor` instead of the `QueryTransformer`. This provides JSON Schema validation, confidence scoring, and access to the extraction resilience pipeline (grounding, recovery, enum normalization). The extractor produces structured intent with `text_queries`, an optional `scope` field (`focused`/`broad`/`exact`), and an optional `output_style` field (`conversational`/`structured`/`hybrid`) that feeds into the synthesis style resolution cascade.

```yaml
intent:
  mode: extract
  extraction_config:
    provider: ollama
    model: qwen3:8b            # Non-thinking model for extraction
    temperature: 0.0            # Deterministic
  num_queries: 3
  domain_context: "OAuth 2.0"
```

When `extraction_config` is absent, the `QueryTransformer` text-parsing path is used (backward compatible). The `extraction_config` dict follows the same shape as wizard `extraction_config` (`provider`, `model`, `temperature`, etc.) and creates a dedicated `SchemaExtractor` via `SchemaExtractor.from_env_config()`.

**Extraction grounding**

When using `extraction_config`, optional extracted fields (`output_style`, `scope`) are automatically grounded against the user's message before being accepted. Grounding uses the standalone utility from `dataknobs_llm.extraction.grounding` — the same type-dispatched checks used by wizard extraction grounding (enum word-boundary match, string word overlap, etc.).

If an optional field is not grounded (the literal enum value does not appear in the user's message), it is dropped from the extraction result and the resolution cascade falls through to config/session/default values. This prevents extraction models from over-classifying — for example, phi4-mini classifying ambiguous queries as `output_style: "structured"` when the user never asked for structured output.

Required fields (`text_queries`) are never dropped by grounding. Per-field behavior can be tuned via `x-extraction` annotations on the schema (e.g., `grounding: "skip"` to bypass grounding for a specific field).

**Optional: Ambiguous query enrichment**

When users ask vague follow-up questions ("Show me an example", "Tell me more about this"), the `ContextualExpander` can enrich the query with keywords from conversation history before passing it to the LLM:

```yaml
intent:
  mode: extract
  expand_ambiguous_queries: true    # Enable for vague follow-ups
  max_context_turns: 3             # How many turns of history to mine
  include_assistant_context: false  # Include assistant messages in mining
```

This is lightweight (no LLM call) and only triggers when `is_ambiguous_query()` detects an ambiguous message. It extracts top-5 keywords (stop-word filtered, deduplicated) from recent user turns and prepends them.

`max_context_turns` controls both the expander's keyword extraction window and the conversation context window passed to the LLM for query generation (each turn maps to ~2 messages).

### Static Mode

Intent is fully defined in configuration. No LLM call. Useful when the config author knows exactly what to search for.

```yaml
intent:
  mode: static
  text_queries:
    - "OAuth 2.0 grant types"
    - "authorization code flow"
  include_message_as_query: true  # Also search using the user's message
  scope: focused
  filters:
    knowledge_base:
      category: "authorization"
```

### Template Mode

A Jinja2 template produces a YAML dict parsed into a `RetrievalIntent`. Template variables include `message` (user message) and `metadata` (conversation metadata). No LLM call.

```yaml
intent:
  mode: template
  template: |
    text_queries:
      - "{{ message }}"
      {% if metadata.get('topic') %}
      - "{{ metadata['topic'] }} overview"
      - "{{ metadata['topic'] }} security considerations"
      {% endif %}
    scope: focused
```

### Default Filters (All Modes)

Config-defined constraints merged into every intent, regardless of mode. These cannot be overridden by the LLM or template:

```yaml
intent:
  mode: extract
  default_filters:
    knowledge_base:
      category: "security"
    case_db:
      status: "published"
```

## Retrieval

Retrieval executes the resolved intent against all configured sources.

### Configuration

```yaml
retrieval:
  top_k: 5                # Maximum results per query per source
  score_threshold: 0.3     # Minimum relevance score to include
  merge_adjacent: true     # Merge adjacent chunks by heading path
  deduplicate: true        # Deduplicate across sources by (source_name, source_id)
```

`score_threshold` applies to all source types. Vector KB sources use semantic similarity scores; database sources use term-coverage scoring (fraction of query terms found in searchable fields, with 2x weight for the content field).

### Sources

Sources can be injected programmatically or constructed from config.

**Automatic (from knowledge_base config):**

When the bot config includes a `knowledge_base` section, the strategy automatically wraps it as a `VectorKnowledgeSource`. No additional configuration needed.

**Config-driven construction:**

```yaml
reasoning:
  strategy: grounded
  sources:
    - type: vector_kb
      name: docs
      weight: 3              # 3 results per round-robin cycle (default: 1)
    - type: database
      name: case_studies
      weight: 1              # 1 result per cycle
      backend: memory
      content_field: summary
      text_search_fields: [title, summary, tags]
      schema:
        fields:
          - name: title
            type: string
          - name: summary
            type: text
```

**Programmatic injection:**

```python
from dataknobs_data.sources.base import GroundedSource

strategy.add_source(my_custom_source)
strategy.set_knowledge_base(my_knowledge_base)  # Wraps as VectorKnowledgeSource
```

### Multi-Source Behavior

When multiple sources are configured:

1. Each source receives the same `RetrievalIntent` and queries independently
2. Results are merged via **weighted round-robin** — each source contributes `weight` results per cycle (default 1), giving higher-weighted sources proportionally more representation
3. Results are deduplicated by `(source_name, source_id)` when `deduplicate: true` (default)
4. Vector KB results are merged by heading path via `ChunkMerger`; other results pass through as-is

Example: with `docs` (weight 3) and `case_studies` (weight 1), the merged order is: docs, docs, docs, case_studies, docs, docs, docs, case_studies, ...

### Custom Sources

Implement the `GroundedSource` ABC from `dataknobs-data`:

```python
from dataknobs_data.sources.base import GroundedSource, RetrievalIntent, SourceResult

class ElasticsearchSource(GroundedSource):
    @property
    def name(self) -> str:
        return "es_docs"

    @property
    def source_type(self) -> str:
        return "elasticsearch"

    async def query(self, intent, *, top_k=5, score_threshold=0.0):
        results = []
        for q in intent.text_queries:
            hits = await self._client.search(q, size=top_k)
            results.extend([
                SourceResult(
                    content=hit["_source"]["content"],
                    source_id=hit["_id"],
                    source_name=self.name,
                    source_type=self.source_type,
                    relevance=hit["_score"],
                )
                for hit in hits
                if hit["_score"] >= score_threshold
            ])
        return sorted(results, key=lambda r: r.relevance, reverse=True)[:top_k]
```

## Topic Index Retrieval

For structured documents, navigating the heading hierarchy or embedding clusters is more reliable than generating search queries via LLM. Topic indices attach to individual sources and provide structure-aware retrieval that replaces or supplements standard vector search.

When a source has a topic index, the grounded strategy uses it instead of standard `text_queries`:

```
for each source:
    if source.topic_index exists:
        results = topic_index.resolve(user_message, llm=..., intent=...)
    else:
        results = source.query(intent, top_k=...)
```

Sources without topic indices continue using standard retrieval. Both approaches coexist in the same pipeline.

### Topic Index Types

| Type | Package | LLM Required | Best For |
|------|---------|-------------|----------|
| `heading_tree` | `dataknobs-bots` | Optional (disambiguation) | Structured documents with heading hierarchy (markdown, RFCs, specs) |
| `cluster` | `dataknobs-data` | No (deterministic) | Any vectorized content — groups by embedding similarity |

### HeadingTreeIndex

Uses heading metadata on chunks (`headings`, `heading_levels`) to identify and expand topic regions. Three entry strategies seed the heading identification:

- **`both`** (default): Merge seeds from heading-text matching AND vector search. Covers both vocabulary-aligned and semantic-gap queries.
- **`heading_match`**: Text-match query terms against heading labels. Avoids the "vector search prefers generic content" problem.
- **`vector`**: Vector search as seed, expand from hit metadata. Bridges vocabulary gaps (e.g. "safety" -> security sections).

All strategies expand matched headings to include descendant chunks — "10. Security Considerations" expands to 10.1 through 10.16.

```yaml
reasoning:
  strategy: grounded
  sources:
    - type: vector_kb
      name: rfc_docs
      topic_index:
        type: heading_tree
        entry_strategy: both          # both, heading_match, vector
        seed_score_threshold: 0.3     # Drop weak vector seeds
        seed_max_results: 10          # Cap seeds before expansion
        min_heading_depth: 1          # Skip title heading (depth 0)
        expansion_mode: subtree       # subtree, children, leaves
        max_expansion_depth: ~        # null = unlimited
        max_expanded_results: 50      # Final cap after expansion
        # Optional LLM heading selection
        resolution_prompt: >
          Given these document sections, select the ones most relevant
          to the user's question. Return only section numbers.
        max_headings_for_llm: 100
        # Heading-text matching configuration
        heading_match:
          min_word_length: 2          # Minimum word length for matching
          min_heading_depth: 1        # Exclude shallow headings
          # stopwords: [custom, list]  # Override default stopwords
        # Per-scope parameter overrides
        scope_profiles:
          focused:
            expansion_mode: children
            max_expansion_depth: 1
          broad:
            expansion_mode: subtree
```

**Expansion modes** control which descendants to include when a heading is matched:

| Mode | Includes | Best For |
|------|----------|----------|
| `subtree` (default) | All descendants at every level | Comprehensive — nothing missed |
| `children` | Immediate children only | Survey — one chunk per subtopic |
| `leaves` | Deepest nodes only | Maximum detail, no structural overhead |

`max_expansion_depth` limits how many levels below the matched heading to traverse. Interacts with `expansion_mode`: `subtree` + `max_expansion_depth: 2` means "all descendants, but only 2 levels deep."

### ClusterTopicIndex

Clusters chunks by embedding similarity at construction time. At query time, embeds the user query and matches to cluster centroids — purely deterministic, no LLM needed.

```yaml
reasoning:
  strategy: grounded
  sources:
    - type: vector_kb
      name: faq_docs
      topic_index:
        type: cluster
        cluster_threshold: 0.7       # Similarity threshold for merging
        min_cluster_size: 2           # Minimum chunks per cluster
        top_clusters: 3               # Max clusters to expand per query
        max_results_per_cluster: 20   # Max chunks per matched cluster
        max_total_results: 50         # Final cap
        centroid_score_threshold: 0.2 # Min centroid similarity
        # Auto-label configuration
        label_min_word_length: 3      # Min word length for labels
        label_top_terms: 3            # Terms per auto-label
        # label_stopwords: [custom]   # Override default stopwords
        scope_profiles:
          focused:
            top_clusters: 1
          broad:
            top_clusters: 5
```

`ClusterTopicIndex` requires an embedding function at query time to embed the user's query for centroid matching. When used via the grounded strategy, the source's existing embedding pipeline provides this automatically.

### Scope Profiles

Both topic index types support **scope profiles** — config-defined parameter sets keyed by the resolved `scope` value from intent extraction. The `scope` field on `RetrievalIntent` captures query intent (`focused`, `broad`, `exact`) via LLM extraction.

Resolution cascade (highest to lowest priority):

1. **Explicit overrides** in `intent.raw_data["topic_index"]` — for template intent mode or custom code paths.
2. **Scope profile** matching `intent.scope` — profile values override config defaults.
3. **Config defaults** — static per-source values.

### Metadata Introspection

`VectorStore.metadata_fields()` returns the set of metadata field names present across stored vectors. This enables auto-detection of whether heading metadata is available for topic-index construction:

```python
fields = await vector_store.metadata_fields()
if {"headings", "heading_levels"} <= fields:
    # Heading metadata available — HeadingTreeIndex viable
    ...
```

Currently implemented by `MemoryVectorStore`. Other backends raise `NotImplementedError` by default.

## Synthesis

Synthesis generates the response from retrieved context.

### Synthesis Styles

Three runtime-switchable synthesis styles control how results are presented:

| Style | Method | Best For |
|-------|--------|----------|
| `conversational` | LLM synthesis (default) | Cross-section reasoning, audience adaptation, follow-up interpretation |
| `structured` | Template with provenance | Research/verification, speed-sensitive, high-trust source content, audit |
| `hybrid` | LLM synthesis + provenance appendix | Both interpretation and source verification |

```yaml
synthesis:
  style: conversational          # or "structured" or "hybrid"
  require_citations: true        # (conversational/hybrid) Cite sources
  allow_parametric: false        # false, true, or "bridge"
  citation_format: section       # "section" (heading paths) or "source" (file paths)
  instruction: >                 # Optional domain-specific synthesis guidance
    Prioritize content that directly addresses the user's question.
  provenance_template: |         # Optional custom template for structured/hybrid output
    {% for r in results %}...{% endfor %}
```

**Conversational** (default) — LLM synthesizes a natural-language response grounded in retrieved results. When `allow_parametric: false`, the LLM explicitly states when KB content is insufficient rather than filling gaps. When `true`, it may supplement but must distinguish KB-grounded claims from general knowledge. When `"bridge"`, the LLM may connect concepts across retrieved content but must not introduce external facts.

**Structured** — A Jinja2 template formats results deterministically. No LLM call. When no custom `template` or `provenance_template` is configured, a built-in default template is used that shows results grouped by source with headings and relevance scores.

**Hybrid** — Runs LLM synthesis, then appends the provenance template output as a source appendix. The appendix uses `provenance_template` (or the built-in default).

### Style Resolution Cascade

The effective synthesis style for each turn is resolved via a priority cascade:

1. **Per-turn** — `output_style` from intent extraction (extract mode with `extraction_config` only). The extraction model defaults to `conversational` and only classifies as `structured` when the user explicitly asks for raw sources or a listing (e.g., "show me the sources", "list the relevant sections"). The classification prompt can be tuned via `intent.output_style_hint` (see below).
2. **Session-level** — `manager.metadata["synthesis_style"]`. Set during scoping or mid-conversation.
3. **Config-level** — `synthesis.style` field.
4. **Legacy mode** — `mode: template` maps to `structured`; `mode: llm` maps to `conversational`.
5. **Default** — `conversational`.

### Tuning Per-Turn Style Classification

The built-in `output_style` schema description tells the extraction model to strongly prefer `conversational`. If the default is too aggressive or too conservative for your model or domain, override it with `output_style_hint`:

```yaml
intent:
  mode: extract
  extraction_config:
    provider: ollama
    model: phi4-mini
  output_style_hint: >
    Always use 'conversational' unless the user explicitly says
    'show me the raw text' or 'list the sources'.
```

When `output_style_hint` is `null` (default), the built-in description is used.

### Custom Templates

Both `structured` and `hybrid` styles accept custom Jinja2 templates:

```yaml
synthesis:
  style: structured
  template: |
    ## Results for: {{ message }}

    {% for result in results %}
    ### {{ result.source_name }} ({{ "%.0f"|format(result.relevance * 100) }}% match)
    {{ result.text }}

    {% endfor %}
    {% if not results %}
    No relevant results found in the knowledge base.
    {% endif %}
```

Template variables: `results` (list of result dicts), `results_by_source` (dict), `context` (formatted context string), `message` (user message), `metadata` (conversation metadata), `intent` (resolved intent dict).

Use `template` for full custom output (overrides the built-in default entirely) or `provenance_template` for just the provenance section (used by hybrid's appendix and as the structured default).

### Legacy Mode Configuration

The `mode` field (`llm` / `template`) continues to work for backward compatibility:

```yaml
# Legacy — equivalent to style: conversational
synthesis:
  mode: llm

# Legacy — equivalent to style: structured
# (uses built-in provenance template if no custom template set)
synthesis:
  mode: template
  template: "..."   # Optional — built-in default used if absent
```

When `style` is set, it takes precedence over `mode`.

## Result Processing Pipeline

An optional post-retrieval processing pipeline runs between merge and synthesis, transforming raw results into ranked, filtered, and optionally clustered output. Configure via `result_processing`:

```yaml
result_processing:
  normalize_strategy: min_max     # Cross-source score normalization
  relative_threshold: 0.5         # Drop results below 50% of best score
  min_results: 3                  # Never drop below this count
  query_rerank_weight: 0.3        # Blend original query relevance
  cluster_strategy: tfidf         # Cluster by TF-IDF similarity
  cluster_threshold: 0.5          # Intra-cluster similarity threshold
  cluster_min_size: 2             # Minimum results to form a cluster
```

### Level 1: Cross-Source Scoring (no embedding dependency)

- **Normalization** (`normalize_strategy`) — Make scores comparable across sources. Strategies: `min_max`, `z_score`, `rank`.
- **Relative filtering** (`relative_threshold`) — Drop results significantly weaker than the best match.
- **Query re-ranking** (`query_rerank_weight`) — Boost results whose content matches the user's original query terms.

### Level 2-3: Clustering + Query-Cluster Scoring

Clustering groups related results and scores each cluster against the user's query:

| Strategy | Requires embeddings | Characteristics |
|----------|-------------------|-----------------|
| `term_overlap` | No | Shared-term grouping. Lightest, fully deterministic. |
| `tfidf` | No | TF-IDF cosine similarity. Good quality, deterministic. |
| `embedding` | Yes | Semantic similarity via embedding model. Highest quality. |

When clustering is active, results are formatted with `<cluster>` XML tags showing label and query relevance. This pairs well with `allow_parametric: "bridge"`, which instructs the LLM to synthesize across clusters.

### Strategy Chains

Every processing stage supports explicit fallback chains:

```yaml
result_processing:
  cluster_strategy:
    - method: embedding
      embedding: {provider: ollama, model: nomic-embed-text}
    - method: tfidf    # fallback if embedding unavailable
```

Strategies are tried in order; `StrategyUnavailable` triggers the next alternative. A single strategy with no alternatives means failure is not silently handled.

## Provenance

When `store_provenance: true` (default), every turn records detailed provenance in `manager.metadata`:

```python
# Current turn's provenance
provenance = manager.metadata["retrieval_provenance"]

# Structure:
{
    "intent": {
        "mode": "resolved",
        "text_queries": ["OAuth grant types", "authorization code"],
        "filters": {},
        "scope": "focused",
        "raw_data": {...},  # Full extraction dict (extract mode)
    },
    "results_by_source": {
        "knowledge_base": [
            {
                "source_id": "chunk_42",
                "source_type": "vector_kb",
                "relevance": 0.92,
                "text": "The authorization code grant type...",
                "text_preview": "The authorization code...",
                "metadata": {"heading_path": "4.1 > Authorization Code"},
            }
        ]
    },
    "results": [...],          # Flat merged list
    "total_results": 8,        # Before dedup
    "deduplicated_to": 5,      # After dedup
    "retrieval_time_ms": 45.2,
    "intent_resolution_time_ms": 120.5,
}

# Full history across turns
history = manager.metadata["retrieval_provenance_history"]  # list of dicts
```

## Query Model Separation

The query-generation LLM can be different from the main synthesis LLM. This allows using a smaller/faster model for generating search queries.

**With `extraction_config` (preferred):**

```yaml
reasoning:
  strategy: grounded
  intent:
    mode: extract
    extraction_config:
      provider: ollama
      model: qwen3:8b       # Smaller, non-thinking model
      temperature: 0.0
```

This creates a dedicated `SchemaExtractor` with its own provider. Avoids using a thinking model (which can spend 2000+ tokens on `<think>` reasoning before producing 3 query strings) for what should be a fast extraction task.

**Without `extraction_config` (legacy, programmatic injection):**

```python
# Inject a separate provider for query generation
bot.reasoning_strategy.set_provider("grounded_query", fast_provider)
```

When neither `extraction_config` nor a separate query provider is set, the bot's main LLM is used for both query generation and synthesis. In this fallback path, `suppress_thinking` is automatically enabled on the `QueryTransformer`, which passes `options: {think: false}` to the LLM provider — preventing thinking models from spending their full token budget on reasoning before producing short query strings.

## Auto-Context Behavior

When a bot has both `knowledge_base` and `strategy: grounded` configured, `from_config()` automatically disables `auto_context` (the KB auto-injection into every message). The grounded strategy handles all KB retrieval structurally — auto-context is redundant and can cause issues with thinking models (oversized prompts from double retrieval).

## Streaming

`stream_chat()` works with the grounded strategy:

1. Intent resolution runs to completion (fast — typically one LLM call or none)
2. Retrieval runs to completion (fast — local vector search)
3. Synthesis streams in real-time (LLM mode) or yields a single chunk (template mode)

```python
async for chunk in bot.stream_chat("What are OAuth grant types?", context):
    print(chunk.content, end="")
```

## Greeting Support

The grounded strategy supports bot-initiated greetings via `greeting_template`:

```yaml
reasoning:
  strategy: grounded
  greeting_template: >
    Hello! I'm a research assistant specializing in {{ domain_context }}.
    What would you like to explore?
```

## Composition API

`GroundedReasoning` exposes a public API designed for composition by other strategies (e.g., `HybridReasoning`) and pipeline consumers. These methods allow composing grounded retrieval and synthesis without reaching into private internals.

### `retrieve_context()` — Retrieval with Optional Pre-Resolved Intent

```python
context, provenance = await grounded.retrieve_context(manager, llm)

# Or skip intent resolution by passing a pre-resolved intent:
from dataknobs_data.sources.base import RetrievalIntent

intent = RetrievalIntent(text_queries=["specific query"], scope="focused")
context, provenance = await grounded.retrieve_context(manager, llm, intent=intent)
```

When `intent` is provided, the intent resolution phase is skipped entirely — useful for pipeline consumers that resolve intent separately (e.g., with an extended schema or profile-specific classification step between intent and retrieval).

### `build_synthesis_system_prompt()` — Prompt with Config Override

```python
prompt = grounded.build_synthesis_system_prompt(kb_context, base_prompt)

# Or override synthesis settings per-turn without mutating strategy config:
from dataknobs_bots.reasoning.grounded_config import GroundedSynthesisConfig

override = GroundedSynthesisConfig(require_citations=False, allow_parametric="bridge")
prompt = grounded.build_synthesis_system_prompt(
    kb_context, base_prompt, synthesis_config=override,
)
```

### `resolve_synthesis()` — Pre-Built Prompt and Config Override

```python
plan = grounded.resolve_synthesis(context, manager, provenance)

# Pass a pre-built prompt to avoid rebuilding it internally:
plan = grounded.resolve_synthesis(
    context, manager, provenance, system_prompt=augmented_prompt,
)

# Or pass a config override (forwarded to build_synthesis_system_prompt):
plan = grounded.resolve_synthesis(
    context, manager, provenance, synthesis_config=override,
)
```

When `system_prompt` is provided, `resolve_synthesis()` skips internal prompt construction. This eliminates double prompt construction when the caller has already built the prompt (e.g., `HybridReasoning` builds it for context injection, then passes it through to post-ReAct synthesis).

### `store_provenance()` — Public Static Method

```python
# Store provenance using the standard metadata key convention:
GroundedReasoning.store_provenance(manager, provenance)

# Sets manager.metadata["retrieval_provenance"] (current turn)
# Appends to manager.metadata["retrieval_provenance_history"] (all turns)
```

This is a `@staticmethod` — callable from any consumer without a strategy instance. `HybridReasoning` uses this to store merged provenance (KB results + tool executions) through a single code path.

## Full Configuration Reference

```yaml
reasoning:
  strategy: grounded

  # Intent resolution
  intent:
    mode: extract              # "extract", "static", or "template"

    # Extract mode
    num_queries: 3             # Number of queries to generate (extract)
    domain_context: ""         # Domain hint for query gen prompt (extract)
    use_conversation_context: true  # Include history in query gen (extract)
    extraction_config: null    # Optional: SchemaExtractor config (extract)
    #   provider: ollama       #   Provider name
    #   model: qwen3:8b        #   Non-thinking model recommended
    #   temperature: 0.0       #   Deterministic extraction
    expand_ambiguous_queries: false  # Enrich vague queries (extract)
    max_context_turns: 3       # History window for enrichment (extract)
    include_assistant_context: false  # Include assistant msgs (extract)

    # Static mode
    text_queries: []           # Fixed queries (static)
    filters: {}                # Fixed filters keyed by source name (static)
    scope: focused             # Retrieval scope (static/template)
    include_message_as_query: true  # Append user message (static)

    # Template mode
    template: null             # Jinja2 template string (template)

    # All modes
    default_filters: {}        # Config-defined constraints (all modes)

  # Retrieval
  retrieval:
    top_k: 5                   # Max results per query per source
    score_threshold: 0.3       # Minimum relevance score (all source types)
    merge_adjacent: true       # Merge adjacent chunks by heading
    deduplicate: true          # Deduplicate by (source_name, source_id)

  # Synthesis
  synthesis:
    mode: llm                  # "llm" or "template"
    require_citations: true    # Instruct LLM to cite (llm mode)
    allow_parametric: false    # false, true, or "bridge"
    citation_format: section   # "section" or "source" (llm mode)
    template: null             # Jinja2 template string (template mode)
    instruction: null          # Optional domain-specific synthesis guidance

  # Result processing (optional — post-retrieval pipeline)
  result_processing:
    normalize_strategy: null   # "min_max", "z_score", "rank", or chain
    relative_threshold: null   # Drop below fraction of best (0.0-1.0)
    min_results: 3             # Floor for filtering
    query_rerank_weight: null  # Query term overlap blend (0.0-1.0)
    cluster_strategy: null     # "term_overlap", "tfidf", "embedding", or chain
    cluster_threshold: 0.7     # Intra-cluster similarity
    cluster_min_size: 2        # Minimum results per cluster

  # Sources (optional — config-driven construction)
  sources:
    - type: vector_kb          # "vector_kb" or "database"
      name: knowledge_base
      weight: 1                # Round-robin weight (default: 1)
      topic_index:             # Optional — per-source topic index
        type: heading_tree     # "heading_tree" or "cluster"
        # HeadingTreeIndex options:
        entry_strategy: both   # "both", "heading_match", "vector"
        seed_score_threshold: 0.3
        seed_max_results: 10
        min_heading_depth: 1
        expansion_mode: subtree  # "subtree", "children", "leaves"
        max_expansion_depth: ~   # null = unlimited
        max_expanded_results: 50
        heading_match:           # Heading-text matching config
          min_word_length: 2
          min_heading_depth: 1
        scope_profiles: {}
        # ClusterTopicIndex options (when type: cluster):
        # cluster_threshold: 0.7
        # min_cluster_size: 2
        # top_clusters: 3
        # max_results_per_cluster: 20
        # max_total_results: 50
        # centroid_score_threshold: 0.2
    - type: database
      name: case_db
      weight: 1                # Higher = more results per cycle
      backend: memory
      content_field: summary
      text_search_fields: [title, summary]

  # Provenance
  store_provenance: true       # Record provenance per turn

  # Greeting
  greeting_template: null      # Optional Jinja2 greeting template
```

## Testing

Use `GroundedConfigBuilder` for fluent config construction, `EchoProvider` for
scripted LLM responses, and `BotTestHarness` for integration tests:

```python
from dataknobs_bots.testing import BotTestHarness, GroundedConfigBuilder
from dataknobs_llm.testing import text_response

config = (
    GroundedConfigBuilder()
    .intent(mode="extract", num_queries=2)
    .retrieval(top_k=3)
    .synthesis(require_citations=True)
    .build()
)

async with await BotTestHarness.create(
    bot_config=config,
    main_responses=[
        text_response("query one\nquery two"),       # Query generation
        text_response("Based on the KB content..."),  # Synthesis
    ],
) as harness:
    result = await harness.chat("What are OAuth grant types?")
    assert "KB content" in result.response
```

For the `extraction_config` path, use `scripted_schema_extractor()` and `set_extractor()`:

```python
from dataknobs_llm.testing import scripted_schema_extractor

extractor, ext_provider = scripted_schema_extractor([
    '{"text_queries": ["OAuth grant types", "auth code flow"], "scope": "focused"}',
])
strategy.set_extractor(extractor)
```

For testing the query transformer and expander independently, see the `test_query.py` tests. For testing custom `GroundedSource` implementations, see `test_grounded_sources.py` in `dataknobs-data`.

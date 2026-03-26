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
  allow_parametric: false        # (conversational/hybrid) Supplement with general knowledge
  citation_format: section       # "section" (heading paths) or "source" (file paths)
  provenance_template: |         # Optional custom template for structured/hybrid output
    {% for r in results %}...{% endfor %}
```

**Conversational** (default) — LLM synthesizes a natural-language response grounded in retrieved results. When `allow_parametric: false`, the LLM explicitly states when KB content is insufficient rather than filling gaps. When `true`, it may supplement but must distinguish KB-grounded claims from general knowledge.

**Structured** — A Jinja2 template formats results deterministically. No LLM call. When no custom `template` or `provenance_template` is configured, a built-in default template is used that shows results grouped by source with headings and relevance scores.

**Hybrid** — Runs LLM synthesis, then appends the provenance template output as a source appendix. The appendix uses `provenance_template` (or the built-in default).

### Style Resolution Cascade

The effective synthesis style for each turn is resolved via a priority cascade:

1. **Per-turn** — `output_style` from intent extraction (extract mode with `extraction_config` only). The LLM infers style from the user's phrasing ("show me the sources" → `structured`, "explain this" → `conversational`).
2. **Session-level** — `manager.metadata["synthesis_style"]`. Set during scoping or mid-conversation.
3. **Config-level** — `synthesis.style` field.
4. **Legacy mode** — `mode: template` maps to `structured`; `mode: llm` maps to `conversational`.
5. **Default** — `conversational`.

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

## Hybrid Composition (Future)

The `retrieve_context()` method is public, designed for future hybrid strategy composition:

```python
# A future HybridReasoning could call this first
context, provenance = await grounded_strategy.retrieve_context(manager, llm)

# Then enter a ReAct loop with the grounded context available
```

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
    allow_parametric: false    # Allow general knowledge (llm mode)
    citation_format: section   # "section" or "source" (llm mode)
    template: null             # Jinja2 template string (template mode)

  # Sources (optional — config-driven construction)
  sources:
    - type: vector_kb          # "vector_kb" or "database"
      name: knowledge_base
      weight: 1                # Round-robin weight (default: 1)
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

Use `EchoProvider` for scripted LLM responses and `BotTestHarness` for integration tests:

```python
from dataknobs_bots.testing import BotTestHarness
from dataknobs_llm.testing import text_response

async with await BotTestHarness.create(
    bot_config={
        "llm": {"provider": "echo", "model": "test"},
        "reasoning": {
            "strategy": "grounded",
            "intent": {"mode": "extract", "num_queries": 2},
            "retrieval": {"top_k": 3},
            "synthesis": {"require_citations": True},
        },
    },
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

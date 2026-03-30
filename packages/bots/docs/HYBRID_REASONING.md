# Hybrid Reasoning Strategy

The hybrid reasoning strategy combines **mandatory KB retrieval** (grounded) with **optional tool execution** (ReAct) in a single pipeline. Every turn retrieves from configured data sources, then the LLM decides whether to use tools for non-KB operations.

## When to Use Hybrid vs. Other Strategies

| Strategy | Use When |
|----------|----------|
| **Hybrid** | Responses MUST be grounded in KB content AND the bot needs tool capabilities (calculations, API calls, document generation). |
| **Grounded** | Responses MUST be grounded in KB content with provenance, but no tools are needed. |
| **ReAct** | Flexible multi-step reasoning with diverse tools. KB search is one of many optional tools (not guaranteed). |
| **Simple** | Direct LLM conversation without tools or retrieval. |
| **Wizard** | Structured data collection with FSM-driven stage progression. |

## Pipeline

```
User message + history
  -> [Grounded Phase] -- deterministic KB retrieval (always executes)
     +-- Intent resolution (extract / static / template)
     +-- Multi-source KB retrieval
     +-- Result processing + formatting
  -> [Context Injection] -- KB context added to system prompt
  -> [ReAct Phase] -- LLM decides whether/which tools to call
     +-- LLM sees: system prompt + KB context + tool definitions
     +-- Tool execution loop (max iterations, duplicate detection)
     +-- Falls back to simple completion if no tools/calls
  -> [Post-ReAct Synthesis] -- optional template formatting
  -> [Provenance Merge] -- KB retrieval + tool executions stored in metadata
```

When no tools are registered, the ReAct phase degrades to a simple KB-augmented LLM completion -- effectively "grounded retrieval + direct answer."

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
  strategy: hybrid
  grounded:
    intent:
      mode: extract
      num_queries: 3
    retrieval:
      top_k: 5
      score_threshold: 0.3
  react:
    max_iterations: 5
  store_provenance: true
```

### Full Configuration

```yaml
reasoning:
  strategy: hybrid

  grounded:
    intent:
      mode: extract          # "extract" (LLM), "static", or "template"
      num_queries: 3
      domain_context: "OAuth 2.0 authorization framework"
      extraction_config:     # optional: dedicated model for intent extraction
        provider: ollama
        model: phi4-mini
    retrieval:
      top_k: 5
      score_threshold: 0.3
    synthesis:
      style: conversational  # "conversational" (default), "structured", "hybrid"
      require_citations: true
      allow_parametric: false
    sources:
      - type: vector_kb
        name: docs
    store_provenance: true   # note: only hybrid-level flag is effective

  react:
    max_iterations: 5
    verbose: false
    store_trace: false

  store_provenance: true     # controls provenance storage for hybrid
  greeting_template: "Hello! I can search our knowledge base and use tools."
```

## Configuration Reference

### `grounded` Section

The grounded section configures the mandatory KB retrieval phase. All settings from the Grounded Reasoning Strategy are supported:

- `intent` -- Intent resolution (extract/static/template modes)
- `retrieval` -- Retrieval parameters (top_k, score_threshold)
- `synthesis` -- Post-ReAct synthesis formatting (see below)
- `sources` -- Data source configuration
- `result_processing` -- Normalization, filtering, re-ranking

### `react` Section

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_iterations` | int | 5 | Maximum ReAct tool-use iterations |
| `verbose` | bool | false | Enable debug-level logging for ReAct steps |
| `store_trace` | bool | false | Store ReAct reasoning trace in metadata |

### Hybrid-Level Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `store_provenance` | bool | true | Record merged provenance in metadata |
| `greeting_template` | string | null | Jinja2 template for bot-initiated greetings |

## Synthesis Styles

The hybrid strategy supports the same synthesis styles as grounded reasoning, applied **after** the ReAct phase completes. The merged provenance (KB results + tool executions) is available to templates.

| Style | Behavior |
|-------|----------|
| `conversational` (default) | ReAct produces the final LLM response directly. No post-processing. |
| `structured` | Provenance template renders the combined results. ReAct LLM response is discarded. |
| `hybrid` | ReAct LLM narrative + provenance template appended. |

### Template Variables

Templates configured via `synthesis.provenance_template` receive:

- `results_by_source` -- KB retrieval results grouped by source
- `results` -- Flat list of all retrieval results
- `context` -- Formatted KB context string
- `message` -- User message
- `metadata` -- Conversation metadata
- `intent` -- Resolved retrieval intent
- `tool_executions` -- List of tool execution records (name, parameters, result, duration)

### Example: Hybrid Style with Tool Results

```yaml
reasoning:
  strategy: hybrid
  grounded:
    synthesis:
      style: hybrid
      provenance_template: |
        ## Sources
        {% for source, results in results_by_source.items() %}
        - **{{ source }}**: {{ results|length }} results
        {% endfor %}
        {% if tool_executions %}
        ## Tool Results
        {% for te in tool_executions %}
        - **{{ te.tool_name }}**: {{ te.result }}
        {% endfor %}
        {% endif %}
```

## Provenance

When `store_provenance: true`, each turn stores provenance in `manager.metadata` using the same schema as the grounded strategy:

- `retrieval_provenance` -- Current turn's provenance dict (overwritten each turn)
- `retrieval_provenance_history` -- Append-only list of all turns' provenance records

The provenance dict includes:

```python
{
    "intent": {"text_queries": [...], "scope": "focused", ...},
    "results_by_source": {"docs": [...]},
    "retrieval_time_ms": 45.8,
    "intent_resolution_time_ms": 150.2,
    "tool_executions": [
        {
            "tool_name": "calculator",
            "parameters": {"expression": "6 * 7"},
            "result": "42",
            "duration_ms": 12.3,
        },
    ],
}
```

## Streaming Behavior

| Scenario | Behavior |
|----------|----------|
| No tools registered | True token streaming via `manager.stream_complete()` |
| Tools registered, no tool calls | Buffered ReAct loop (single iteration), yielded as one chunk |
| Tools registered, tool calls occur | Full ReAct loop (buffered), final response yielded |
| Structured synthesis style | Template output yielded as single chunk (no LLM streaming) |
| Hybrid synthesis style (no tools) | LLM stream + template appendix as final chunk |

## Architecture

`HybridReasoning` composes two child strategy instances via delegation:

- `GroundedReasoning` -- Owned, handles the retrieval phase via `retrieve_context()`
- `ReActReasoning` -- Owned, handles the tool execution phase via `generate()`

Neither child is visible to `DynaBot` -- `HybridReasoning` is the sole registered strategy. Provider injection, lifecycle management (`close()`), and tool execution forwarding are delegated to both children.

### Context Injection

Retrieved KB context is injected into the system prompt via `GroundedReasoning.build_synthesis_system_prompt()`, which includes grounding instructions (citation format, parametric knowledge policy, custom instructions) from the grounded config.

### Tool Execution Forwarding

Tool executions from the ReAct phase are collected via `get_and_clear_tool_executions()` and forwarded through `HybridReasoning._tool_executions`, enabling DynaBot's `on_tool_executed` middleware hooks to fire correctly.

# Performance & Benchmarking

Optimize LLM applications with benchmarking tools and caching strategies.

## Overview

Performance features include:

- **Benchmarking Framework**: Measure and track performance
- **RAG Caching**: Query hash-based cache matching
- **Jinja2 Optimization**: Compiled templates for faster rendering
- **Async Support**: Full async/await throughout
- **Performance Profiling**: Identify bottlenecks

## RAG Caching

### Conversation-Level Caching

```python
from dataknobs_llm.conversations import ConversationManager

# Enable RAG caching
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    cache_rag_results=True,      # Store RAG metadata
    reuse_rag_on_branch=True     # Reuse when branching
)

# Add message with RAG
await manager.add_message(
    role="user",
    prompt_name="code_question",  # Has RAG configured
    params={"language": "python"}
)
# RAG searches executed once, metadata stored

# Inspect RAG metadata
rag_info = await manager.get_rag_metadata()
print(f"Query: {rag_info['RAG_CONTENT']['query']}")
print(f"Retrieved {len(rag_info['RAG_CONTENT']['results'])} docs")

# Branch conversation - RAG is reused
await manager.switch_to_node("0")
await manager.complete(branch_name="alternative")
# Cached RAG used, saving time and cost
```

### Cache Matching

RAG caching uses SHA256 query hashing:

```python
# Same query → cache hit
query1 = "python decorators"
query2 = "python decorators"  # Same hash, cache reused

# Different query → cache miss
query3 = "python generators"  # Different hash, new search
```

## Template Optimization

### Jinja2 Compilation

Jinja2 automatically compiles templates to bytecode:

```python
from dataknobs_llm.prompts import AsyncPromptBuilder

# Templates are compiled on first use
builder = AsyncPromptBuilder(library=library)

# First render: Compiles template
result1 = await builder.render_user_prompt("greeting", params)

# Subsequent renders: Uses compiled template (faster)
result2 = await builder.render_user_prompt("greeting", params)
```

### Caching Prompt Libraries

```python
# ✅ Reuse library instances
library = FileSystemPromptLibrary(prompt_dir=Path("prompts/"))
builder = AsyncPromptBuilder(library=library)

# Use same builder for multiple renders
for params in param_list:
    result = await builder.render_user_prompt("template", params)

# ❌ Don't recreate for each use
for params in param_list:
    library = FileSystemPromptLibrary(...)  # Slow!
    builder = AsyncPromptBuilder(library=library)  # Slow!
    result = await builder.render_user_prompt("template", params)
```

## Async Performance

### Parallel RAG Searches

```python
# Multiple RAG searches execute in parallel
builder = AsyncPromptBuilder(
    library=library,
    adapters={
        'docs': docs_adapter,
        'examples': examples_adapter,
        'api': api_adapter
    }
)

# All three RAG searches run concurrently
result = await builder.render_user_prompt(
    'multi_rag_prompt',  # Uses all three adapters
    params=params
)
```

### Batching Requests

```python
import asyncio

# Process multiple prompts concurrently
async def process_batch(prompts, params_list):
    tasks = [
        builder.render_user_prompt(prompt, params)
        for prompt, params in zip(prompts, params_list)
    ]
    return await asyncio.gather(*tasks)

results = await process_batch(prompts, params_list)
```

## Detailed Documentation

Comprehensive performance documentation is available in the package:

### Local Package Documentation

The LLM package includes detailed documentation in `packages/llm/docs/`:

- **rag-caching.md** - RAG metadata caching, query hashing, and cache configuration
- **best-practices.md** - Performance best practices and optimization patterns

These files are available in the source package at `packages/llm/docs/` or in the [GitHub repository](https://github.com/kbs-labs/dataknobs/tree/main/packages/llm/docs)

## Performance Metrics

### Key Metrics

1. **Rendering Time**: Template rendering performance
   - Target: < 10ms for simple templates
   - Measured: Mean, median, std dev

2. **RAG Search Time**: Search performance
   - Target: < 100ms for cached queries
   - Measured: Per-adapter performance

3. **Conversation Operations**: Add message, complete, branch
   - Target: < 50ms for add message
   - Measured: Operation latency

4. **Memory Usage**: RAM consumption
   - Monitored: Library loading, conversation trees
   - Optimized: Lazy loading, cleanup

## Optimization Tips

### 1. Use Appropriate RAG k Values

```python
# ✅ Use only what you need
rag_configs:
  - adapter_name: docs
    k: 3  # Top 3 results usually sufficient

# ❌ Don't over-retrieve
rag_configs:
  - adapter_name: docs
    k: 50  # Slow and often unnecessary
```

### 2. Enable Caching Strategically

```python
# ✅ Enable for conversation apps
manager = await ConversationManager.create(
    ...,
    cache_rag_results=True,
    reuse_rag_on_branch=True
)

# ❌ Don't enable for one-shot queries
# (overhead not worth it)
```

### 3. Use Jinja2 Mode for Complex Templates

```python
# ✅ For complex logic, use pure Jinja2
template_mode: "jinja2"
template: |
  {% for item in items %}
    {% if item.priority > 5 %}
      High: {{item.name}}
    {% endif %}
  {% endfor %}

# Faster than multiple conditional blocks
```

### 4. Lazy Load Prompt Libraries

```python
# For large libraries, consider lazy loading
class LazyLibrary(AbstractPromptLibrary):
    def get_system_prompt(self, name):
        # Load on demand
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name]
```

### 5. Profile Your Application

```python
import cProfile
import pstats

# Profile prompt rendering
profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = await builder.render_user_prompt("template", params)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

## Benchmark Results

Typical performance on modern hardware:

| Operation | Ops/sec | Mean Time |
|-----------|---------|-----------|
| Simple rendering | ~100,000 | ~0.01ms |
| Conditional rendering | ~50,000 | ~0.02ms |
| Complex template | ~10,000 | ~0.1ms |
| RAG search (cached) | ~10,000 | ~0.1ms |
| RAG search (uncached) | ~100 | ~10ms |
| Add message | ~1,000 | ~1ms |
| Complete (LLM call) | ~10 | ~100ms |

## Monitoring


### Real-Time Metrics

```python
import time

# Measure prompt rendering
start = time.perf_counter()
result = await builder.render_user_prompt("template", params)
duration = time.perf_counter() - start

metrics.histogram("prompt.render.duration", duration * 1000)
```

## See Also

- [Conversation Management](conversations.md) - RAG caching in conversations
- [Prompt Engineering](prompts.md) - Template optimization
- [API Reference](../api/prompts.md) - Prompts API (includes benchmarking utilities)
- [Examples](../examples/advanced-prompting.md) - Advanced prompting patterns

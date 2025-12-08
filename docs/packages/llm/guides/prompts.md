# Prompt Engineering Guide

This guide covers the comprehensive prompt engineering system in dataknobs-llm.

## Overview

The prompt system provides:

- **Template System**: Jinja2-powered templates with 50+ filters
- **Conditional Logic**: Smart conditionals with `(( ))` syntax
- **RAG Integration**: Explicit placeholder-based retrieval
- **Validation**: Parameter validation at ERROR/WARN/IGNORE levels
- **Library Management**: Filesystem, config, and composite libraries

## Quick Reference

### Basic Template Rendering

```python
from dataknobs_llm.prompts import render_template

# Simple rendering
result = render_template(
    "Hello {{name}}!",
    {"name": "Alice"}
)

# With Jinja2 filters
result = render_template(
    "Hello {{name|upper}}!",
    {"name": "alice"}
)
# Result: "Hello ALICE!"
```

### Using Prompt Libraries

```python
from dataknobs_llm.prompts import FileSystemPromptLibrary
from pathlib import Path

# Load prompts from directory
library = FileSystemPromptLibrary(
    prompt_dir=Path("prompts/")
)

# Get a prompt template
template = library.get_system_prompt("code_analysis")
print(template['template'])
print(template['defaults'])
```

### RAG Integration

```python
from dataknobs_llm.prompts import (
    AsyncPromptBuilder,
    AsyncDictResourceAdapter
)

# Create RAG adapter
docs_adapter = AsyncDictResourceAdapter({
    "doc1": {"content": "Python basics..."},
    "doc2": {"content": "Advanced topics..."},
})

# Build prompt with RAG
builder = AsyncPromptBuilder(
    library=library,
    adapters={'docs': docs_adapter}
)

result = await builder.render_user_prompt(
    'code_question',
    params={'language': 'python', 'topic': 'decorators'}
)
```

## Detailed Documentation

For comprehensive documentation on the prompt system, additional resources are available in the package:

### Local Package Documentation

The LLM package includes detailed documentation in `packages/llm/docs/`:

- **USER_GUIDE.md** - Complete user guide covering template syntax, libraries, RAG, and validation
- **JINJA2_INTEGRATION.md** - Comprehensive Jinja2 features guide with 50+ filters
- **JINJA2_MIGRATION.md** - Migration guide for transitioning to Jinja2
- **BEST_PRACTICES.md** - Best practices for prompt engineering

These files are available in the source package at `packages/llm/docs/` or in the [GitHub repository](https://github.com/kbs-labs/dataknobs/tree/main/packages/llm/docs)

## Inline Prompts

For quick prototyping or one-off prompts, you can render inline content directly without defining templates in a library:

```python
from dataknobs_llm.prompts import AsyncPromptBuilder, ConfigPromptLibrary

library = ConfigPromptLibrary()  # Can be empty
builder = AsyncPromptBuilder(library=library)

# Render inline system prompt with template variables
result = await builder.render_inline_system_prompt(
    content="You are a helpful {{role}} assistant.",
    params={"role": "coding"}
)
print(result.content)  # "You are a helpful coding assistant."

# Render inline user prompt
result = await builder.render_inline_user_prompt(
    content="Help me understand {{topic}}",
    params={"topic": "decorators"}
)
```

### Inline Prompts with RAG

Inline prompts also support RAG enhancement by providing `rag_configs`:

```python
# Setup adapter
docs_adapter = AsyncDictResourceAdapter({
    "guidelines": {"content": "Be helpful and concise."},
})

builder = AsyncPromptBuilder(
    library=library,
    adapters={"docs": docs_adapter}
)

# Inline system prompt with RAG
result = await builder.render_inline_system_prompt(
    content="You are a helpful assistant.\n\nGuidelines:\n{{GUIDELINES}}",
    rag_configs=[{
        "adapter_name": "docs",
        "query": "assistant guidelines",
        "placeholder": "GUIDELINES",
        "k": 3,
        "header": "",
        "item_template": "- {{content}}\n"
    }]
)
```

This is particularly useful when:
- Prototyping prompts before adding to a library
- Creating one-off prompts that don't need versioning
- Dynamically constructing prompts at runtime
- Testing RAG integration quickly

## Common Patterns

### 1. Multi-RAG Prompts

```python
# Define prompt with multiple RAG sources
"""
Context from documentation:
{{RAG_DOCS}}

Context from examples:
{{RAG_EXAMPLES}}

Question: {{question}}
"""

# Configure in YAML
rag_configs:
  - adapter_name: docs
    query: "{{topic}}"
    k: 3
    placeholder: "RAG_DOCS"
  - adapter_name: examples
    query: "{{topic}} examples"
    k: 2
    placeholder: "RAG_EXAMPLES"
```

### 2. Conditional Sections

```python
# Optional sections based on parameters
template = """
Analyze this {{language}} code((, focusing on {{focus}})):

{{code}}

((
Performance notes: {{performance_notes}}
))
"""
```

### 3. Template Inheritance

```yaml
# base.yaml
template: |
  You are a helpful assistant.
  {{content}}

# specific.yaml
extends: "base"
sections:
  content: |
    Analyze this code: {{code}}
```

## Template Syntax Quick Reference

| Feature | Syntax | Example |
|---------|--------|---------|
| Variable | `{{var}}` | `{{name}}` |
| Filter | `{{var|filter}}` | `{{name|upper}}` |
| Conditional | `((content))` | `((, age {{age}})` |
| Jinja2 If | `{% if %}` | `{% if admin %}Admin{% endif %}` |
| Jinja2 For | `{% for %}` | `{% for item in items %}{{item}}{% endfor %}` |
| Include | `{% include %}` | `{% include 'header' %}` |
| Placeholder | `{{RAG_*}}` | `{{RAG_CONTENT}}` |

## RAG Adapters

### Built-in Adapters

1. **DictResourceAdapter**: In-memory dictionary-based
2. **AsyncDictResourceAdapter**: Async version
3. **InMemoryAdapter**: Simple in-memory adapter
4. **DataknobsBackendAdapter**: Dataknobs backend integration

### Custom Adapters

```python
from dataknobs_llm.prompts import AsyncResourceAdapter

class MyCustomAdapter(AsyncResourceAdapter):
    async def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        # Implement custom search logic
        results = await my_search_function(query, k)
        return [
            {
                "content": r.text,
                "metadata": r.meta,
                "score": r.relevance
            }
            for r in results
        ]
```

## Validation

### Validation Levels

```python
from dataknobs_llm.prompts import ValidationLevel

# ERROR: Raise exception for missing params
builder = AsyncPromptBuilder(
    library=library,
    validation_level=ValidationLevel.ERROR
)

# WARN: Log warnings (default)
builder = AsyncPromptBuilder(
    library=library,
    validation_level=ValidationLevel.WARN
)

# IGNORE: Silent
builder = AsyncPromptBuilder(
    library=library,
    validation_level=ValidationLevel.IGNORE
)
```

### Template Validation

```yaml
# In YAML prompt
validation:
  required_params:
    - language
    - code
  optional_params:
    - style_guide
    - max_length
```

## Performance Tips

1. **Use compiled templates**: Jinja2 compiles templates to bytecode
2. **Cache prompt libraries**: Reuse library instances
3. **Limit RAG results**: Use appropriate `k` values
4. **Enable RAG caching**: For conversation-level caching

```python
# Enable RAG caching
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    cache_rag_results=True,
    reuse_rag_on_branch=True
)
```

## Testing Prompts

```python
import pytest
from dataknobs_llm.prompts import render_template

def test_greeting_template():
    result = render_template(
        "Hello {{name}}!",
        {"name": "Alice"}
    )
    assert result == "Hello Alice!"

def test_conditional():
    result = render_template(
        "Hello{{(, {{age}} years old)}}!",
        {"age": 30}
    )
    assert ", 30 years old" in result
```

## See Also

- [Conversation Management](conversations.md) - Using prompts in conversations
- [Versioning & A/B Testing](versioning.md) - Prompt version control
- [Performance & Benchmarking](performance.md) - Optimization
- [API Reference](../api/prompts.md) - Complete API documentation

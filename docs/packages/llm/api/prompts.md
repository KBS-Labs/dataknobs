# Prompts API

Template library, builders, and RAG integration for structured prompt engineering.

## Overview

The prompts API provides a flexible system for managing, rendering, and versioning prompt templates with support for Jinja2 templating and RAG integration.

## Prompt Library

### Abstract Interface

::: dataknobs_llm.prompts.AbstractPromptLibrary
    options:
      show_source: true
      heading_level: 3
      members:
        - get_system_prompt
        - get_user_prompt
        - list_prompts

### Implementations

#### FileSystemPromptLibrary

::: dataknobs_llm.prompts.FileSystemPromptLibrary
    options:
      show_source: true
      heading_level: 4

#### ConfigPromptLibrary

::: dataknobs_llm.prompts.ConfigPromptLibrary
    options:
      show_source: true
      heading_level: 4

## Prompt Builders

### PromptBuilder (Sync)

::: dataknobs_llm.prompts.PromptBuilder
    options:
      show_source: true
      heading_level: 3
      members:
        - render_system_prompt
        - render_user_prompt
        - render_prompt

### AsyncPromptBuilder

::: dataknobs_llm.prompts.AsyncPromptBuilder
    options:
      show_source: true
      heading_level: 3
      members:
        - render_system_prompt
        - render_user_prompt
        - render_prompt

## Resource Adapters

Resource adapters provide data for RAG and template variables.

### DictResourceAdapter

::: dataknobs_llm.prompts.DictResourceAdapter
    options:
      show_source: true
      heading_level: 3

### InMemoryAdapter

::: dataknobs_llm.prompts.InMemoryAdapter
    options:
      show_source: true
      heading_level: 3

### DataknobsBackendAdapter

::: dataknobs_llm.prompts.DataknobsBackendAdapter
    options:
      show_source: true
      heading_level: 3

## Validation

### ValidationLevel

::: dataknobs_llm.prompts.ValidationLevel
    options:
      show_source: true
      heading_level: 3

### ValidationConfig

::: dataknobs_llm.prompts.ValidationConfig
    options:
      show_source: true
      heading_level: 3

## Prompt Types

### PromptTemplateDict

::: dataknobs_llm.prompts.PromptTemplateDict
    options:
      show_source: true
      heading_level: 3

### RAGConfig

::: dataknobs_llm.prompts.RAGConfig
    options:
      show_source: true
      heading_level: 3

## Usage Examples

### Basic Template Loading

```python
from dataknobs_llm.prompts import FileSystemPromptLibrary, AsyncPromptBuilder
from pathlib import Path

# Load from filesystem
library = FileSystemPromptLibrary(prompt_dir=Path("prompts/"))

# Create builder
builder = AsyncPromptBuilder(library=library)

# Render prompt
result = await builder.render_user_prompt(
    "code_review",
    params={"language": "python", "code": "def foo(): pass"}
)
print(result)
```

### In-Memory Library

```python
from dataknobs_llm.prompts import InMemoryPromptLibrary, PromptTemplateDict

# Create templates
templates = {
    "greeting": PromptTemplateDict(
        template="Hello {{name}}!",
        defaults={"name": "User"},
        validation={"required": ["name"]}
    )
}

# Create library
library = InMemoryPromptLibrary(prompts={"system": templates})

# Use with builder
builder = AsyncPromptBuilder(library=library)
result = await builder.render_system_prompt("greeting", {"name": "Alice"})
```

### RAG Integration

RAG (Retrieval-Augmented Generation) is configured in prompt templates using RAGConfig:

```yaml
# In your prompt YAML file (e.g., user/code_question.yaml)
template: |
  Answer this question about {{language}}:
  {{question}}

  Relevant documentation:
  {{RAG_DOCS}}

rag_configs:
  - adapter_name: docs
    query_template: "{{language}} programming"
    k: 3
    placeholder: "RAG_DOCS"
```

For comprehensive RAG documentation, including caching and configuration, see:
- **Location**: `packages/llm/docs/RAG_CACHING.md`
- **Location**: `packages/llm/docs/USER_GUIDE.md` (RAG section)

### Template Modes

```python
# String formatting mode
template = PromptTemplateDict(
    template_mode="string",
    template="Hello {name}!"
)

# Jinja2 mode (default)
template = PromptTemplateDict(
    template_mode="jinja2",
    template="Hello {{name | upper}}!"
)

# Conditional mode
template = PromptTemplateDict(
    template_mode="conditional",
    template="Hello {{name}}!",
    conditional_blocks=[
        {
            "condition": "is_premium",
            "content": "Welcome to premium support!"
        }
    ]
)
```

### Validation

```python
# Define validation rules
template = PromptTemplateDict(
    template="Code review for {{language}}: {{code}}",
    validation={
        "required": ["language", "code"],
        "types": {
            "language": "str",
            "code": "str"
        }
    }
)

# Validation happens automatically during render
try:
    result = await builder.render_user_prompt(
        "code_review",
        params={"language": "python"}  # Missing 'code'
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

### Jinja2 Filters

```python
# Use built-in filters
template = """
Name: {{name | upper}}
Date: {{date | format_date('%Y-%m-%d')}}
List: {{items | join(', ')}}
JSON: {{data | tojson}}
"""

result = await builder.render_user_prompt(
    "example",
    params={
        "name": "alice",
        "date": datetime.now(),
        "items": ["a", "b", "c"],
        "data": {"key": "value"}
    }
)
```

## Prompt File Format

### YAML Format

```yaml
# system/code_reviewer.yaml
template: |
  You are an expert code reviewer.
  Focus on {{language}} best practices.

  Review the following code:
  {{code}}

defaults:
  language: python

validation:
  required:
    - code
  types:
    language: str
    code: str

rag_configs:
  - adapter_name: docs
    query_template: "{{language}} best practices"
    k: 5
    placeholder: "BEST_PRACTICES"

metadata:
  version: "1.0.0"
  author: "team"
```

## See Also

- [Prompt Engineering Guide](../guides/prompts.md) - Detailed guide
- [LLM API](../../../api/dataknobs-llm.md) - LLM provider interface
- [Versioning API](versioning.md) - Version management
- [Examples](../examples/advanced-prompting.md) - Advanced examples

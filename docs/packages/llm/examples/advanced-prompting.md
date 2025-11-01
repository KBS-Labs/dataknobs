# Advanced Prompting Examples

Complex templates, RAG integration, and advanced patterns.

## Jinja2 Templates

### Template with Filters

```python
from dataknobs_llm.prompts import InMemoryPromptLibrary, AsyncPromptBuilder, PromptTemplate

# Template using Jinja2 filters
template = PromptTemplate(
    template_mode="jinja2",
    template="""
    Analyze this {{language | upper}} code:

    ```{{language}}
    {{code | trim}}
    ```

    Focus on:
    {% for topic in focus_areas %}
    - {{topic | title}}
    {% endfor %}

    Date: {{date | format_date('%Y-%m-%d')}}
    """,
    defaults={
        "language": "python",
        "focus_areas": ["performance", "readability"]
    }
)

library = InMemoryPromptLibrary(prompts={"user": {"code_analysis": template}})
builder = AsyncPromptBuilder(library=library)

result = await builder.render_user_prompt(
    "code_analysis",
    params={
        "code": "  def hello():  \n    print('hi')  ",
        "date": datetime.now()
    }
)
print(result)
```

### Conditional Blocks

```python
template = PromptTemplate(
    template_mode="jinja2",
    template="""
    {% if user_level == 'beginner' %}
    Explain in simple terms:
    {% elif user_level == 'intermediate' %}
    Provide a detailed explanation:
    {% else %}
    Give an expert-level analysis:
    {% endif %}

    {{topic}}

    {% if include_examples %}
    Include code examples.
    {% endif %}
    """,
    defaults={
        "user_level": "intermediate",
        "include_examples": True
    }
)

result = await builder.render_user_prompt(
    "explanation",
    params={
        "topic": "Python decorators",
        "user_level": "beginner",
        "include_examples": True
    }
)
```

### Loops and Lists

```python
template = PromptTemplate(
    template_mode="jinja2",
    template="""
    Review the following {{language}} files:

    {% for file in files %}
    ## File {{loop.index}}: {{file.name}}
    {{file.content}}

    {% if not loop.last %}---{% endif %}
    {% endfor %}

    Provide feedback on:
    {{feedback_areas | join(', ')}}
    """,
    validation={
        "required": ["files"],
        "types": {"files": "list"}
    }
)

result = await builder.render_user_prompt(
    "multi_file_review",
    params={
        "language": "python",
        "files": [
            {"name": "main.py", "content": "def main(): pass"},
            {"name": "utils.py", "content": "def helper(): pass"}
        ],
        "feedback_areas": ["structure", "naming", "documentation"]
    }
)
```

## RAG Integration

### Basic RAG Configuration

RAG (Retrieval-Augmented Generation) is configured in prompt templates using YAML:

```yaml
# user/qa_with_docs.yaml
template: |
  Answer this question about {{topic}}:
  {{question}}

  Relevant documentation:
  {{RAG_DOCS}}

rag_configs:
  - adapter_name: docs
    query_template: "{{topic}} {{question}}"
    k: 3
    placeholder: "RAG_DOCS"
```

Then use the prompt:

```python
result = await builder.render_user_prompt(
    "qa_with_docs",
    params={
        "topic": "Python",
        "question": "How do decorators work?"
    }
)
# RAG results automatically retrieved and injected at {{RAG_DOCS}}
```

**Note**: RAG adapters are provided through FSM resource integration. See `packages/llm/docs/RAG_CACHING.md` for comprehensive RAG documentation.

### Multiple RAG Sources

```yaml
# user/comprehensive_qa.yaml
template: |
  Question: {{question}}

  Documentation:
  {{RAG_DOCS}}

  Code Examples:
  {{RAG_EXAMPLES}}

  API Reference:
  {{RAG_API}}

rag_configs:
  - adapter_name: docs
    query_template: "{{question}}"
    k: 3
    placeholder: "RAG_DOCS"

  - adapter_name: examples
    query_template: "{{question}} code example"
    k: 2
    placeholder: "RAG_EXAMPLES"

  - adapter_name: api
    query_template: "{{question}} API"
    k: 2
    placeholder: "RAG_API"
```

Usage:

```python
result = await builder.render_user_prompt(
    "comprehensive_qa",
    params={"question": "How to use async/await?"}
)
```

### Dynamic RAG Queries

```python
# Template with dynamic RAG query
# template: |
#   {% if include_examples %}
#   Show me {{language}} code for {{task}}.
#
#   Similar examples:
#   {{RAG_EXAMPLES}}
#   {% else %}
#   Explain how to {{task}} in {{language}}.
#
#   Reference:
#   {{RAG_DOCS}}
#   {% endif %}
#
# rag_configs:
#   - adapter_name: examples
#     query_template: "{{language}} {{task}} example"
#     k: 3
#     placeholder: "RAG_EXAMPLES"
#     condition: "include_examples"
#
#   - adapter_name: docs
#     query_template: "{{language}} {{task}}"
#     k: 5
#     placeholder: "RAG_DOCS"
#     condition: "not include_examples"

result = await builder.render_user_prompt(
    "flexible_qa",
    params={
        "language": "python",
        "task": "parse JSON",
        "include_examples": True
    }
)
```

## Complex Validation

### Type Validation

```python
template = PromptTemplate(
    template="Analyze {{metric}} for {{dates | length}} dates",
    validation={
        "required": ["metric", "dates"],
        "types": {
            "metric": "str",
            "dates": "list"
        },
        "constraints": {
            "metric": {"min_length": 1},
            "dates": {"min_length": 1, "max_length": 31}
        }
    }
)

# Valid
result = await builder.render_user_prompt(
    "analysis",
    params={
        "metric": "revenue",
        "dates": ["2024-01-01", "2024-01-02"]
    }
)

# Invalid - raises ValidationError
try:
    result = await builder.render_user_prompt(
        "analysis",
        params={"metric": "revenue"}  # Missing 'dates'
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

### Custom Validators

```python
from dataknobs_llm.prompts import AbstractPromptLibrary

class ValidatingPromptLibrary(AbstractPromptLibrary):
    def validate_params(self, template, params):
        # Custom validation logic
        if "code" in params:
            code = params["code"]
            if len(code) > 10000:
                raise ValueError("Code too long (max 10,000 chars)")
            if "eval(" in code or "exec(" in code:
                raise ValueError("Dangerous code detected")

        return super().validate_params(template, params)

library = ValidatingPromptLibrary()
builder = AsyncPromptBuilder(library=library)
```

## Template Composition

### Nested Templates

```python
# Base template
base_template = PromptTemplate(
    template="""
    You are {{role}}.
    {{content}}
    """,
    defaults={"role": "a helpful assistant"}
)

# Specialized template
code_review_template = PromptTemplate(
    template="""
    {% include 'base' %}

    Review this {{language}} code:
    {{code}}
    """,
    defaults={"language": "python"}
)

# Template inheritance
class InheritingLibrary(InMemoryPromptLibrary):
    def __init__(self):
        super().__init__(prompts={
            "system": {
                "base": base_template,
                "code_review": code_review_template
            }
        })

    def get_system_prompt(self, name):
        template = super().get_system_prompt(name)
        if "{% include 'base' %}" in template.template:
            base = super().get_system_prompt("base")
            template.template = template.template.replace(
                "{% include 'base' %}",
                base.template
            )
        return template
```

### Macros

```python
template = PromptTemplate(
    template_mode="jinja2",
    template="""
    {% macro format_code(language, code) %}
    ```{{language}}
    {{code | trim}}
    ```
    {% endmacro %}

    Review these files:

    {% for file in files %}
    ## {{file.name}}
    {{ format_code(file.language, file.code) }}
    {% endfor %}
    """,
    validation={"required": ["files"]}
)
```

## Dynamic Prompts

### Prompt Generation Based on Context

```python
async def generate_dynamic_prompt(context):
    """Generate prompt based on runtime context."""

    if context["complexity"] == "high":
        template_name = "expert_analysis"
        params = {
            "depth": "comprehensive",
            "include_diagrams": True
        }
    elif context["complexity"] == "medium":
        template_name = "detailed_analysis"
        params = {
            "depth": "moderate",
            "include_examples": True
        }
    else:
        template_name = "simple_analysis"
        params = {
            "depth": "basic",
            "language": "simple"
        }

    params.update(context["data"])

    return await builder.render_user_prompt(template_name, params)

# Use it
context = {
    "complexity": "high",
    "data": {
        "topic": "distributed systems",
        "subtopics": ["consistency", "availability", "partition tolerance"]
    }
}

prompt = await generate_dynamic_prompt(context)
```

### Adaptive Templates

```python
class AdaptivePromptBuilder:
    def __init__(self, builder, llm):
        self.builder = builder
        self.llm = llm
        self.user_preferences = {}

    async def render_adaptive(self, template_name, params, user_id):
        # Get user preferences
        prefs = self.user_preferences.get(user_id, {})

        # Adjust parameters based on preferences
        if prefs.get("expertise") == "expert":
            params["detail_level"] = "high"
            params["use_jargon"] = True
        else:
            params["detail_level"] = "basic"
            params["use_jargon"] = False

        # Render with adjusted params
        return await self.builder.render_user_prompt(template_name, params)

    async def learn_preference(self, user_id, interaction):
        """Learn from user interactions."""
        # Analyze interaction to infer preferences
        if "more detail" in interaction.lower():
            self.user_preferences.setdefault(user_id, {})
            self.user_preferences[user_id]["detail_level"] = "high"

adaptive_builder = AdaptivePromptBuilder(builder, llm)
```

## Performance Optimization

### Template Caching

```python
from functools import lru_cache

class CachedPromptBuilder(AsyncPromptBuilder):
    @lru_cache(maxsize=128)
    def _compile_template(self, template_str):
        """Cache compiled templates."""
        return self.jinja_env.from_string(template_str)

    async def render_prompt(self, template, params):
        # Use cached compilation
        compiled = self._compile_template(template.template)
        return await super().render_prompt(template, params)
```

### RAG Caching

The LLM package provides conversation-level RAG caching:

```python
from dataknobs_llm.conversations import ConversationManager

# Enable RAG caching
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    cache_rag_results=True,      # Cache RAG metadata
    reuse_rag_on_branch=True     # Reuse when branching
)

# First message with RAG - executes search
await manager.add_message(
    role="user",
    prompt_name="qa_with_docs",
    params={"question": "How do decorators work?"}
)

# Branch conversation - reuses cached RAG results
await manager.switch_to_node("0")
await manager.complete(branch_name="alternative")
```

See `packages/llm/docs/RAG_CACHING.md` for details.

### Parallel RAG Searches

```python
import asyncio

async def parallel_rag_render(builder, template_name, params_list):
    """Render multiple prompts with RAG in parallel."""
    tasks = [
        builder.render_user_prompt(template_name, params)
        for params in params_list
    ]
    return await asyncio.gather(*tasks)

# Process batch
params_list = [
    {"question": "What is Python?"},
    {"question": "What is JavaScript?"},
    {"question": "What is Ruby?"}
]

results = await parallel_rag_render(builder, "qa_with_docs", params_list)
```

## Error Handling

### Graceful Degradation

```python
async def render_with_fallback(builder, template_name, params):
    """Render with fallback on RAG failure."""
    try:
        return await builder.render_user_prompt(template_name, params)
    except RAGError:
        # Fallback to non-RAG version
        print("RAG failed, using fallback template")
        return await builder.render_user_prompt(
            f"{template_name}_no_rag",
            params
        )

result = await render_with_fallback(builder, "qa_with_docs", params)
```

### Template Debugging

```python
from dataknobs_llm.prompts import TemplateRenderError

try:
    result = await builder.render_user_prompt("complex_template", params)
except TemplateRenderError as e:
    print(f"Template error: {e}")
    print(f"Template: {e.template_name}")
    print(f"Line: {e.line_number}")
    print(f"Context: {e.context}")
```

## See Also

- [Basic Usage Examples](basic-usage.md) - Getting started
- [Conversation Flow Examples](conversation-flows.md) - FSM workflows
- [A/B Testing Examples](ab-testing.md) - Version management
- [Prompt Engineering Guide](../guides/prompts.md) - Detailed guide

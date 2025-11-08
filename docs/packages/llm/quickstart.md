# Quick Start

Get up and running with the LLM package in 5 minutes.

## Installation

```bash
pip install dataknobs-llm
```

Or with specific LLM provider support:

```bash
# OpenAI
pip install dataknobs-llm[openai]

# Anthropic
pip install dataknobs-llm[anthropic]

# All providers
pip install dataknobs-llm[all]
```

## Basic Usage

### 1. Create an LLM Provider

```python
from dataknobs_llm import create_llm_provider, LLMConfig

# OpenAI
config = LLMConfig(provider="openai", api_key="your-api-key")
llm = create_llm_provider(config)

# Anthropic
config = LLMConfig(provider="anthropic", api_key="your-api-key")
llm = create_llm_provider(config)

# With custom configuration
config = LLMConfig(
    provider="openai",
    api_key="your-api-key",
    model="gpt-4",
    temperature=0.7
)
llm = create_llm_provider(config)
```

### 2. Simple Completion

```python
# Asynchronous (recommended)
response = await llm.acomplete("What is Python?")
print(response.content)

# Or using synchronous provider
config = LLMConfig(provider="openai", api_key="your-api-key")
llm_sync = create_llm_provider(config, is_async=False)
response = llm_sync.complete("What is Python?")
print(response.content)
```

### 3. Structured Prompts

```python
from dataknobs_llm.prompts import FileSystemPromptLibrary, AsyncPromptBuilder
from pathlib import Path

# Load prompts from filesystem
library = FileSystemPromptLibrary(prompt_dir=Path("prompts/"))
builder = AsyncPromptBuilder(library=library)

# Render and use prompts
prompt = await builder.render_user_prompt(
    "code_review",
    params={"language": "python", "code": "def foo(): pass"}
)

response = await llm.acomplete(prompt)
```

### 4. Conversations

```python
from dataknobs_llm.conversations import (
    ConversationManager,
    DataknobsConversationStorage
)
from dataknobs_data.backends import AsyncMemoryDatabase

# Create storage
db = AsyncMemoryDatabase()
storage = DataknobsConversationStorage(db)

# Create conversation
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage
)

# Add user message
await manager.add_message(
    role="user",
    prompt_name="greeting",
    params={"name": "Alice"}
)

# Get assistant response
response = await manager.complete()
print(response.content)

# Continue conversation
await manager.add_message(
    role="user",
    content="Tell me more about Python decorators"
)
response = await manager.complete()
```

## Common Patterns

### RAG (Retrieval-Augmented Generation)

RAG is configured in prompt templates using YAML:

```yaml
# prompts/user/code_question.yaml
template: |
  Answer this {{language}} question:
  {{question}}

  Relevant documentation:
  {{RAG_DOCS}}

rag_configs:
  - adapter_name: docs
    query_template: "{{language}} {{question}}"
    k: 3
    placeholder: "RAG_DOCS"
```

Then use resource adapters to provide the data:

```python
from dataknobs_llm.prompts import InMemoryAdapter

# Create resource adapter with documents
adapter = InMemoryAdapter(
    documents=[
        {"id": "1", "content": "Python is a programming language"},
        {"id": "2", "content": "Python supports decorators"}
    ]
)

# Use in prompt builder
builder = AsyncPromptBuilder(
    library=library,
    adapters={"docs": adapter}
)

# RAG automatically retrieves and injects relevant docs
result = await builder.render_user_prompt(
    "code_question",
    params={"language": "python", "question": "What are decorators?"}
)
```

### A/B Testing

```python
from dataknobs_llm.prompts import (
    VersionManager,
    ABTestManager,
    PromptVariant
)

# Create versions
vm = VersionManager()
v1 = await vm.create_version(
    name="greeting",
    prompt_type="system",
    template="Hello {{name}}!",
    version="1.0.0"
)

v2 = await vm.create_version(
    name="greeting",
    prompt_type="system",
    template="Hi {{name}}, welcome!"
)

# Create A/B test
ab = ABTestManager()
exp = await ab.create_experiment(
    name="greeting",
    prompt_type="system",
    variants=[
        PromptVariant("1.0.0", 0.5, "Control"),
        PromptVariant("1.0.1", 0.5, "Treatment")
    ]
)

# Get variant for user (sticky)
variant = await ab.get_variant_for_user(exp.experiment_id, "user123")
```

## Next Steps

### Learn More

- **[Prompt Engineering Guide](guides/prompts.md)** - Master prompt templates
- **[Conversation Management](guides/conversations.md)** - Multi-turn conversations
- **[Versioning & A/B Testing](guides/versioning.md)** - Track and test prompts
- **[Performance & Benchmarking](guides/performance.md)** - Optimize your application

### Examples

- **[Basic Usage Examples](examples/basic-usage.md)** - Common use cases
- **[Advanced Prompting](examples/advanced-prompting.md)** - Complex templates
- **[Conversation Flows](examples/conversation-flows.md)** - FSM-based workflows
- **[A/B Testing](examples/ab-testing.md)** - Running experiments

### API Reference

- **[LLM API](../../api/dataknobs-llm.md)** - LLM provider interface
- **[Prompts API](api/prompts.md)** - Prompt library and builders
- **[Conversations API](api/conversations.md)** - Conversation management
- **[Versioning API](api/versioning.md)** - Version and experiment management

## Configuration

### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY=your-key

# Anthropic
export ANTHROPIC_API_KEY=your-key

# Prompt directory
export PROMPT_DIR=/path/to/prompts
```

### File Structure

Organize your prompts in a directory structure:

```
prompts/
├── system/
│   ├── greeting.yaml
│   └── code_reviewer.yaml
└── user/
    ├── code_question.yaml
    └── general_question.yaml
```

Example prompt file (`greeting.yaml`):

```yaml
template: |
  You are a friendly assistant.
  Greet the user named {{name}}.

defaults:
  name: User

validation:
  required:
    - name
```

## Troubleshooting

### Common Issues

**Import Error**: Make sure you've installed the package and any required extras:
```bash
pip install dataknobs-llm[openai]
```

**API Key Not Found**: Set environment variables or pass explicitly:
```python
from dataknobs_llm import create_llm_provider, LLMConfig

config = LLMConfig(provider="openai", api_key="your-key")
llm = create_llm_provider(config)
```

**Template Not Found**: Check your prompt directory path:
```python
library = FileSystemPromptLibrary(prompt_dir=Path("prompts/"))
```

### Getting Help

- **Documentation**: [Full Documentation](index.md)
- **GitHub**: [Issues](https://github.com/kbs-labs/dataknobs/issues)
- **Examples**: See the [examples](examples/basic-usage.md) directory

## What's Next?

Now that you have the basics, explore:

1. **Advanced Prompting**: Learn Jinja2 templating, RAG integration, and conditional logic
2. **Conversation Trees**: Branch conversations and explore alternatives
3. **Performance Optimization**: Use benchmarking and caching for production
4. **A/B Testing**: Run experiments to find the best prompts

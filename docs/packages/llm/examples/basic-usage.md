# Basic Usage Examples

Common use cases and patterns for getting started with the LLM package.

## Simple Completions

### Basic Question Answering

```python
from dataknobs_llm import create_llm_provider, LLMConfig

# Create LLM provider
config = LLMConfig(provider="openai", api_key="your-key")
llm = create_llm_provider(config, is_async=False)

# Ask a question
response = llm.complete("What is Python?")
print(response.content)
```

### With System Prompt

```python
# Set system prompt
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I reverse a list in Python?"}
]

response = llm.complete(messages)
print(response.content)
```

### Async Completion

```python
import asyncio
from dataknobs_llm import create_llm_provider, LLMConfig

async def ask_question():
    config = LLMConfig(provider="openai", api_key="your-key")
    llm = create_llm_provider(config)  # is_async=True by default
    response = await llm.acomplete("What are Python decorators?")
    print(response.content)

asyncio.run(ask_question())
```

## Streaming Responses

### Basic Streaming

```python
# Stream response word by word
for chunk in llm.stream("Tell me a story about Python"):
    print(chunk.content, end="", flush=True)
print()  # Newline at end
```

### Async Streaming

```python
from dataknobs_llm import create_llm_provider, LLMConfig

async def stream_story():
    config = LLMConfig(provider="openai", api_key="your-key")
    llm = create_llm_provider(config)
    async for chunk in llm.astream("Tell me a story"):
        print(chunk.content, end="", flush=True)
    print()

asyncio.run(stream_story())
```

### Collecting Stream

```python
# Collect all chunks
chunks = []
for chunk in llm.stream("Count to 10"):
    chunks.append(chunk.content)
    print(chunk.content, end="", flush=True)

full_response = "".join(chunks)
print(f"\n\nFull response: {full_response}")
```

## Template-Based Prompts

### Simple Template

```python
from dataknobs_llm.prompts import InMemoryPromptLibrary, AsyncPromptBuilder, PromptTemplate

# Create template
templates = {
    "greeting": PromptTemplate(
        template="Greet a user named {{name}} in {{language}}.",
        defaults={"language": "English"}
    )
}

# Create library and builder
library = InMemoryPromptLibrary(prompts={"user": templates})
builder = AsyncPromptBuilder(library=library)

# Render prompt
prompt = await builder.render_user_prompt(
    "greeting",
    params={"name": "Alice", "language": "Spanish"}
)

# Use with LLM
response = await llm.acomplete(prompt)
print(response.content)
```

### File-Based Templates

```python
from dataknobs_llm.prompts import FileSystemPromptLibrary
from pathlib import Path

# Create directory structure:
# prompts/
#   user/
#     code_review.yaml

# code_review.yaml content:
# template: |
#   Review this {{language}} code:
#   {{code}}
# defaults:
#   language: python

# Load from filesystem
library = FileSystemPromptLibrary(prompt_dir=Path("prompts/"))
builder = AsyncPromptBuilder(library=library)

# Use template
prompt = await builder.render_user_prompt(
    "code_review",
    params={
        "language": "python",
        "code": "def add(a, b): return a + b"
    }
)

response = await llm.acomplete(prompt)
print(response.content)
```

## Multi-Turn Conversations

### Simple Chat

```python
from dataknobs_llm.conversations import (
    ConversationManager,
    DataknobsConversationStorage
)
from dataknobs_data.backends import AsyncMemoryDatabase

# Setup storage
db = AsyncMemoryDatabase()
storage = DataknobsConversationStorage(db)

# Create conversation
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage
)

# First exchange
await manager.add_message(role="user", content="What is a decorator in Python?")
response = await manager.complete()
print(f"Assistant: {response.content}\n")

# Follow-up
await manager.add_message(role="user", content="Can you show me an example?")
response = await manager.complete()
print(f"Assistant: {response.content}\n")

# Another follow-up
await manager.add_message(role="user", content="Explain the @property decorator")
response = await manager.complete()
print(f"Assistant: {response.content}")
```

### Chat with System Prompt

```python
# Create with system prompt
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    system_prompt_name="python_tutor",  # From your prompt library
    system_prompt_params={"skill_level": "beginner"}
)

# All completions use this system prompt
await manager.add_message(role="user", content="What are list comprehensions?")
response = await manager.complete()
```

### View Conversation History

```python
# Get all messages
history = manager.get_history()

print("Conversation History:")
for msg in history:
    print(f"{msg.role}: {msg.content[:50]}...")
```

## Configuration Examples

### Temperature Control

```python
from dataknobs_llm import create_llm_provider, LLMConfig

# Creative writing (high temperature)
creative_config = LLMConfig(
    provider="openai",
    api_key="your-key",
    temperature=0.9
)
creative_llm = create_llm_provider(creative_config, is_async=False)
story = creative_llm.complete("Write a creative story about a robot")

# Factual answers (low temperature)
factual_config = LLMConfig(
    provider="openai",
    api_key="your-key",
    temperature=0.1
)
factual_llm = create_llm_provider(factual_config, is_async=False)
answer = factual_llm.complete("What is the capital of France?")
```

### Model Selection

```python
from dataknobs_llm import create_llm_provider, LLMConfig

# Use GPT-4
config_gpt4 = LLMConfig(
    provider="openai",
    api_key="your-key",
    model="gpt-4"
)
llm_gpt4 = create_llm_provider(config_gpt4, is_async=False)

# Use GPT-3.5
config_gpt35 = LLMConfig(
    provider="openai",
    api_key="your-key",
    model="gpt-3.5-turbo"
)
llm_gpt35 = create_llm_provider(config_gpt35, is_async=False)

# Use Claude
config_claude = LLMConfig(
    provider="anthropic",
    api_key="your-key",
    model="claude-3-sonnet-20240229"
)
llm_claude = create_llm_provider(config_claude, is_async=False)
```

## Error Handling

### Basic Error Handling

```python
from dataknobs_llm.exceptions import LLMError, RateLimitError, InvalidRequestError

try:
    response = llm.complete("What is Python?")
    print(response.content)
except RateLimitError as e:
    print(f"Rate limit exceeded. Please wait and try again. {e}")
except InvalidRequestError as e:
    print(f"Invalid request: {e}")
except LLMError as e:
    print(f"LLM error occurred: {e}")
```

### Retry Logic

```python
import time

def complete_with_retry(llm, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return llm.complete(prompt)
        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

response = complete_with_retry(llm, "What is Python?")
```

## Token Usage Tracking

### Basic Tracking

```python
response = llm.complete("Explain Python decorators")

print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
```

### Cost Estimation

```python
# GPT-4 pricing (example rates, check OpenAI for current)
PROMPT_COST_PER_1K = 0.03
COMPLETION_COST_PER_1K = 0.06

response = llm.complete("Explain machine learning")

prompt_cost = (response.usage.prompt_tokens / 1000) * PROMPT_COST_PER_1K
completion_cost = (response.usage.completion_tokens / 1000) * COMPLETION_COST_PER_1K
total_cost = prompt_cost + completion_cost

print(f"Estimated cost: ${total_cost:.4f}")
```

### Batch Processing with Budget

```python
def process_with_budget(llm, prompts, max_cost=1.00):
    total_cost = 0.0
    results = []

    for prompt in prompts:
        response = llm.complete(prompt)

        # Calculate cost for this request
        request_cost = (
            (response.usage.prompt_tokens / 1000) * PROMPT_COST_PER_1K +
            (response.usage.completion_tokens / 1000) * COMPLETION_COST_PER_1K
        )

        total_cost += request_cost
        results.append(response.content)

        if total_cost >= max_cost:
            print(f"Budget reached: ${total_cost:.4f}")
            break

    return results, total_cost

prompts = ["Explain Python", "Explain JavaScript", "Explain Ruby"]
results, cost = process_with_budget(llm, prompts, max_cost=0.50)
print(f"Processed {len(results)} prompts for ${cost:.4f}")
```

## Environment Variables

### Using Environment Variables

```python
import os
from dataknobs_llm import create_llm_provider, LLMConfig

# Set in environment
# export OPENAI_API_KEY=your-key

# Read from environment
config = LLMConfig(
    provider="openai",
    api_key=os.getenv("OPENAI_API_KEY")
)
llm = create_llm_provider(config)
```

### Configuration File

```python
# config.yaml
# openai_api_key: your-key
# model: gpt-4
# temperature: 0.7

import yaml
from dataknobs_llm import create_llm_provider, LLMConfig

with open("config.yaml") as f:
    config_data = yaml.safe_load(f)

config = LLMConfig(
    provider="openai",
    api_key=config_data["openai_api_key"],
    model=config_data["model"],
    temperature=config_data["temperature"]
)
llm = create_llm_provider(config)
```

## Batch Processing

### Process Multiple Prompts

```python
async def process_batch(llm, prompts):
    tasks = [llm.acomplete(prompt) for prompt in prompts]
    responses = await asyncio.gather(*tasks)
    return [r.content for r in responses]

prompts = [
    "What is Python?",
    "What is JavaScript?",
    "What is Ruby?"
]

results = await process_batch(llm, prompts)
for i, result in enumerate(results):
    print(f"\nPrompt {i+1}: {prompts[i]}")
    print(f"Response: {result[:100]}...")
```

### Parallel Processing with Rate Limiting

```python
import asyncio
from asyncio import Semaphore

async def process_with_rate_limit(llm, prompts, max_concurrent=5):
    semaphore = Semaphore(max_concurrent)

    async def process_one(prompt):
        async with semaphore:
            return await llm.acomplete(prompt)

    tasks = [process_one(p) for p in prompts]
    responses = await asyncio.gather(*tasks)
    return [r.content for r in responses]

prompts = [f"Tell me about topic {i}" for i in range(20)]
results = await process_with_rate_limit(llm, prompts, max_concurrent=5)
```

## See Also

- [Advanced Prompting Examples](advanced-prompting.md) - Complex templates and RAG
- [Conversation Flow Examples](conversation-flows.md) - FSM-based workflows
- [A/B Testing Examples](ab-testing.md) - Version management and experiments
- [Quick Start Guide](../quickstart.md) - Getting started

# Dataknobs LLM - Best Practices Guide

**Package**: `dataknobs_llm`
**Version**: 0.1.0
**Last Updated**: 2025-10-29

---

## Table of Contents

1. [Prompt Design](#prompt-design)
2. [Template Organization](#template-organization)
3. [Validation Strategy](#validation-strategy)
4. [RAG Best Practices](#rag-best-practices)
5. [Conversation Management](#conversation-management)
6. [Middleware Patterns](#middleware-patterns)
7. [Storage and Performance](#storage-and-performance)
8. [Error Handling](#error-handling)
9. [Testing](#testing)
10. [Production Deployment](#production-deployment)

---

## Prompt Design

### Write Clear, Focused Prompts

**Good**:
```yaml
# system/code_reviewer.yaml
template: |
  You are an expert code reviewer specializing in {{language}}.

  Review code for:
  1. Correctness and bugs
  2. Best practices and style
  3. Security vulnerabilities
  4. Performance issues

  Provide specific, actionable feedback.

defaults:
  language: "Python"
```

**Avoid**:
```yaml
# Too vague, no clear instructions
template: "You are helpful. Do code stuff."
```

### Use Conditional Sections Wisely

**Good** - Optional context:
```yaml
template: |
  Analyze this {{language}} code((, following {{style_guide}})):

  {{code}}

  ((Focus on {{aspect}}.))
```

**Avoid** - Nested conditionals that are hard to read:
```yaml
# Too complex
template: "Hello (({{name}}((, age {{age}}((, from {{city}}((, in {{country}}))))))))."
```

**Recommendation**: Limit conditional nesting to 2 levels maximum.

### Provide Sensible Defaults

**Good**:
```yaml
template: "Analyze {{code}} for {{aspect}}"
defaults:
  aspect: "correctness and style"  # Reasonable default
validation:
  level: "warn"
  required_params: ["code"]  # Only truly required params
```

**Avoid**:
```yaml
# No defaults, everything required
validation:
  level: "error"
  required_params: ["code", "aspect", "language", "style", "format"]
```

**Recommendation**: Make parameters required only if they're truly essential.

### Use Descriptive Variable Names

**Good**:
```yaml
template: "Review {{source_code}} for {{security_issue_types}}"
```

**Avoid**:
```yaml
template: "Review {{x}} for {{y}}"  # Unclear what x and y are
```

### Structure Long Prompts with Sections

**Good**:
```yaml
template: |
  {{ROLE}}

  {{INSTRUCTIONS}}

  {{EXAMPLES}}

  {{INPUT}}

sections:
  ROLE: "You are an expert {{domain}} consultant."
  INSTRUCTIONS: |
    Provide analysis following these steps:
    1. Identify key patterns
    2. Assess quality
    3. Suggest improvements
  EXAMPLES: "((Example: {{example_text}}))"
  INPUT: "Input to analyze:\n{{content}}"
```

**Benefit**: Easier to read, maintain, and override specific sections.

---

## Template Organization

### Directory Structure

**Recommended structure**:
```
prompts/
├── system/           # System prompts (one per file)
│   ├── base_assistant.yaml
│   ├── code_reviewer.yaml
│   └── customer_support.yaml
├── user/             # User prompts (by name)
│   ├── initial_question.yaml
│   ├── detailed_question.yaml
│   ├── submit_code.yaml
│   └── follow_up.yaml
├── rag/              # RAG configurations
│   ├── docs_search.yaml
│   ├── code_search.yaml
│   └── faq_search.yaml
└── README.md         # Documentation
```

### Naming Conventions

**System prompts**: `{purpose}.yaml`
- `code_reviewer.yaml`
- `customer_support.yaml`
- `data_analyst.yaml`

**User prompts**: `{descriptive_name}.yaml`
- `initial_question.yaml` - Basic question
- `detailed_question.yaml` - Detailed question with context
- `submit_code.yaml` - Code submission prompt

**RAG configs**: `{source}_search.yaml`
- `docs_search.yaml`
- `code_search.yaml`

### Use Template Inheritance

**Base template** (system/base_assistant.yaml):
```yaml
template: |
  {{ROLE}}

  {{CAPABILITIES}}

  {{GUIDELINES}}

sections:
  ROLE: "You are a helpful AI assistant."
  CAPABILITIES: "You can answer questions and provide information."
  GUIDELINES: "Be accurate, concise, and helpful."
```

**Specialized template** (system/code_assistant.yaml):
```yaml
extends: "base_assistant"

sections:
  ROLE: "You are an expert software engineer and code reviewer."
  CAPABILITIES: |
    You can:
    - Review code for bugs and best practices
    - Explain programming concepts
    - Suggest improvements and optimizations
```

**Benefits**:
- DRY (Don't Repeat Yourself)
- Consistent structure across prompts
- Easy to update common elements
- Clear hierarchy

### Use Composite Libraries for Environments

**Pattern**:
```python
from dataknobs_llm.prompts import (
    FileSystemPromptLibrary,
    CompositePromptLibrary
)

# Environment-specific overrides
prod_library = FileSystemPromptLibrary("prompts/production/")
base_library = FileSystemPromptLibrary("prompts/base/")

# Production uses specific prompts, falls back to base
library = CompositePromptLibrary(
    libraries=[prod_library, base_library],
    names=["production", "base"]
)
```

**Use cases**:
- Development vs production prompts
- A/B testing variants
- Customer-specific customizations
- Feature flags

---

## Validation Strategy

### Choose Appropriate Validation Levels

**ERROR** - For critical parameters:
```yaml
# system/database_query.yaml
template: "Execute query on {{database}}: {{query}}"
validation:
  level: "error"  # Missing these would cause failures
  required_params: ["database", "query"]
```

**WARN** - For optional but recommended parameters:
```yaml
# system/analysis.yaml
template: "Analyze {{data}}((, focusing on {{aspect}}))"
validation:
  level: "warn"  # Missing aspect is okay, but user should know
  required_params: []
  optional_params: ["aspect"]
```

**IGNORE** - For always-optional parameters:
```yaml
# system/chat.yaml
template: "Chat about (({{topic}}))"
validation:
  level: "ignore"  # Truly optional, no warnings needed
```

### Validation Hierarchy

```python
# 1. Template defines defaults
# prompts/system/analyze.yaml
validation:
  level: "warn"  # Default for this template

# 2. Builder can override
builder = AsyncPromptBuilder(
    library=library,
    default_validation_level=ValidationLevel.ERROR  # Override for all prompts
)

# 3. Runtime can override
result = await builder.render_system_prompt(
    "analyze",
    params={...},
    validation_level=ValidationLevel.IGNORE  # Override for this call
)
```

**Recommendation**: Use template-level validation for most cases, runtime overrides sparingly.

### Document Required Parameters

```yaml
template: "Analyze {{code}} for {{issues}}"

# Clear documentation
validation:
  level: "error"
  required_params: ["code", "issues"]

# Add metadata for tools/docs
metadata:
  description: "Code analysis prompt"
  parameters:
    code:
      type: "string"
      description: "Source code to analyze"
    issues:
      type: "array"
      description: "List of issue types to check"
      examples: ["security", "performance", "style"]
```

---

## RAG Best Practices

### Separate RAG Configs from Prompts

**Good** - Reusable RAG config:
```yaml
# rag/code_search.yaml
adapter_name: "codebase"
query: "{{language}} {{topic}}"
k: 5
filters:
  type: "code"
score_threshold: 0.7

# system/code_helper.yaml
template: |
  Relevant code examples:
  {{RAG_CONTENT}}

  Help with: {{question}}

rag_config_refs: ["code_search"]  # Reference
```

**Benefits**:
- Reuse across multiple prompts
- Easy to test different RAG strategies
- Swap configs at runtime

### Parameterize RAG Queries

**Good**:
```yaml
# rag/docs_search.yaml
query: "{{topic}} {{language}} {{framework}}"
k: 5
```

**Avoid**:
```yaml
# Too specific, not reusable
query: "Python async programming with asyncio"
```

### Set Appropriate k and Thresholds

```yaml
# For broad context
k: 10
score_threshold: 0.5

# For precise results
k: 3
score_threshold: 0.8

# For maximum recall
k: 20
score_threshold: 0.3
```

**Recommendation**: Start with `k=5, score_threshold=0.7` and tune based on results.

### Handle RAG Failures Gracefully

```python
try:
    result = await builder.render_system_prompt(
        "prompt_with_rag",
        params={...},
        include_rag=True
    )
except Exception as e:
    logger.error(f"RAG search failed: {e}")
    # Fallback: render without RAG
    result = await builder.render_system_prompt(
        "prompt_with_rag",
        params={...},
        include_rag=False
    )
```

### Format RAG Content Clearly

**Prompt template**:
```yaml
template: |
  Use these reference documents:

  {{RAG_CONTENT}}

  ---

  Question: {{question}}

  Answer based on the documents above.
```

**Benefit**: Clear separation between RAG content and user input.

---

## Conversation Management

### Design Conversation State Carefully

**Good** - Meaningful metadata:
```python
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    metadata={
        "user_id": user_id,
        "session_id": session_id,
        "conversation_type": "code_review",
        "language": "python",
        "started_at": datetime.now().isoformat()
    }
)
```

**Benefits**:
- Easy querying and filtering
- Analytics and tracking
- Context for resumption

### Use Branching for Exploration

**Pattern**:
```python
# Main conversation path
await manager.add_message(role="user", content="Explain lists")
response1 = await manager.complete()

# Try alternative explanation
await manager.switch_to_node("0")
response2 = await manager.complete(branch_name="simple-explanation")

# Compare approaches
branches = manager.get_branches()
# Pick best branch or show user options
```

**Use cases**:
- A/B testing responses
- Exploring different solutions
- Providing alternatives to users
- Debugging conversation flows

### Implement Conversation Limits

**Pattern**:
```python
async def safe_complete(manager, max_turns=50):
    messages = manager.state.get_current_messages()

    if len(messages) > max_turns:
        # Summarize or warn
        logger.warning(f"Conversation exceeding {max_turns} turns")
        # Consider summarization
        return await summarize_and_continue(manager)

    return await manager.complete()
```

**Recommendation**: Set limits based on:
- Token limits of your LLM
- Cost considerations
- User experience (very long conversations are confusing)

### Clean Up Old Conversations

**Pattern**:
```python
from datetime import datetime, timedelta

async def cleanup_old_conversations(storage, days=30):
    """Delete conversations older than specified days."""
    cutoff = datetime.now() - timedelta(days=days)

    conversations = await storage.list_conversations(limit=1000)

    for conv in conversations:
        if conv.updated_at < cutoff:
            await storage.delete_conversation(conv.conversation_id)
            logger.info(f"Deleted old conversation: {conv.conversation_id}")
```

**Recommendation**: Implement regular cleanup or archival strategy.

### Save Important Checkpoints

**Pattern**:
```python
# After important milestones
await manager.add_message(role="user", content="Approve this solution")
response = await manager.complete()

# Save current state
conv_id = manager.conversation_id
checkpoint_node = manager.state.current_node_id

# Store for later resumption
save_checkpoint(user_id, conv_id, checkpoint_node)
```

---

## Middleware Patterns

### Order Middleware Thoughtfully

**Good order**:
```python
middleware = [
    LoggingMiddleware(logger),          # Log everything first
    ContentFilterMiddleware(...),        # Filter early
    ValidationMiddleware(...),           # Validate after filtering
    MetadataMiddleware(...)              # Add metadata last
]
```

**Reasoning**:
- Logging captures all requests/responses
- Filtering removes problematic content early
- Validation checks cleaned content
- Metadata added after all processing

### Use Validation Middleware Selectively

**Good** - Only for critical conversations:
```python
# Production customer support - validate responses
validation_mw = ValidationMiddleware(
    llm=validation_llm,
    prompt_builder=builder,
    validation_prompt="validate_support_response",
    auto_retry=True
)

# Internal testing - skip validation (save costs)
middleware = [LoggingMiddleware(logger)] if env == "prod" else []
```

**Costs**: ValidationMiddleware doubles LLM calls (one for response, one for validation).

### Implement Custom Middleware for Business Logic

**Example - Rate limiting**:
```python
class RateLimitMiddleware(ConversationMiddleware):
    def __init__(self, max_requests_per_minute=10):
        self.max_requests = max_requests_per_minute
        self.requests = {}  # user_id -> [(timestamp, count)]

    async def process_request(self, messages, state):
        user_id = state.metadata.get("user_id")

        if self._is_rate_limited(user_id):
            raise ValueError("Rate limit exceeded")

        self._record_request(user_id)
        return messages

    async def process_response(self, response, state):
        return response

    def _is_rate_limited(self, user_id):
        # Implementation...
        pass

    def _record_request(self, user_id):
        # Implementation...
        pass
```

**Example - Token counting**:
```python
class TokenBudgetMiddleware(ConversationMiddleware):
    def __init__(self, max_tokens=100000):
        self.max_tokens = max_tokens
        self.used_tokens = 0

    async def process_response(self, response, state):
        if response.usage:
            self.used_tokens += response.usage.get("total_tokens", 0)

            if self.used_tokens > self.max_tokens:
                logger.warning(f"Token budget exceeded: {self.used_tokens}/{self.max_tokens}")

            response.metadata["cumulative_tokens"] = self.used_tokens

        return response
```

### Handle Middleware Errors

**Pattern**:
```python
class SafeValidationMiddleware(ConversationMiddleware):
    async def process_response(self, response, state):
        try:
            # Validation logic
            is_valid = await self._validate(response)

            if not is_valid:
                response.metadata["validation_failed"] = True
                logger.warning("Validation failed")

        except Exception as e:
            # Don't fail entire request on validation error
            logger.error(f"Validation error: {e}")
            response.metadata["validation_error"] = str(e)

        return response
```

**Recommendation**: Decide per-middleware whether errors should:
- Fail the entire request (critical middleware)
- Log and continue (nice-to-have middleware)

---

## Storage and Performance

### Choose Storage Backend Based on Needs

**Development/Testing**:
```python
from dataknobs_data.backends import AsyncMemoryDatabase

storage = DataknobsConversationStorage(AsyncMemoryDatabase())
```
- Fast, no setup
- Data lost on restart
- Perfect for tests

**Local Development**:
```python
from dataknobs_data.backends import AsyncSQLiteDatabase

storage = DataknobsConversationStorage(
    AsyncSQLiteDatabase(db_path="conversations.db")
)
```
- Persistent across restarts
- No external dependencies
- Good for local development

**Production (Small Scale)**:
```python
from dataknobs_data.backends import AsyncSQLiteDatabase

storage = DataknobsConversationStorage(
    AsyncSQLiteDatabase(db_path="/var/data/conversations.db")
)
```
- Simple deployment
- Good for <100k conversations
- Single server setups

**Production (Large Scale)**:
```python
from dataknobs_data.backends import AsyncPostgresDatabase

storage = DataknobsConversationStorage(
    AsyncPostgresDatabase(
        connection_string="postgresql://user:pass@host:5432/db"
    )
)
```
- Scalable to millions of conversations
- ACID guarantees
- Advanced querying
- Suitable for distributed systems

### Implement Caching for Frequent Accesses

**Pattern**:
```python
from functools import lru_cache

class CachedConversationManager:
    def __init__(self, storage, cache_size=100):
        self.storage = storage
        self.cache = {}  # conversation_id -> state

    async def get_conversation(self, conversation_id):
        # Check cache first
        if conversation_id in self.cache:
            return self.cache[conversation_id]

        # Load from storage
        state = await self.storage.load_conversation(conversation_id)

        # Cache it
        if state:
            self.cache[conversation_id] = state

        return state

    async def save_conversation(self, state):
        # Update cache
        self.cache[state.conversation_id] = state

        # Persist to storage
        await self.storage.save_conversation(state)
```

**Benefits**:
- Reduces database load
- Faster conversation access
- Important for high-traffic applications

### Batch Operations When Possible

**Good**:
```python
# Process multiple conversations in parallel
async def process_batch(conversation_ids):
    tasks = [
        process_conversation(conv_id)
        for conv_id in conversation_ids
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

**Avoid**:
```python
# Sequential processing
for conv_id in conversation_ids:
    await process_conversation(conv_id)  # Slow!
```

### Monitor Performance

**Pattern**:
```python
import time

class PerformanceMiddleware(ConversationMiddleware):
    async def process_request(self, messages, state):
        state.metadata["request_start"] = time.time()
        return messages

    async def process_response(self, response, state):
        duration = time.time() - state.metadata.get("request_start", 0)

        response.metadata["llm_duration_seconds"] = duration

        if duration > 5.0:
            logger.warning(f"Slow LLM response: {duration}s")

        return response
```

---

## Error Handling

### Handle Missing Prompts Gracefully

**Pattern**:
```python
from dataknobs_llm.prompts import PromptNotFoundError

async def safe_render(builder, prompt_name, params, fallback_template=None):
    try:
        return await builder.render_user_prompt(prompt_name, params=params)
    except PromptNotFoundError:
        logger.warning(f"Prompt '{prompt_name}' not found")

        if fallback_template:
            # Use fallback
            from dataknobs_llm.prompts import render_template
            return render_template(fallback_template, params)

        raise  # Re-raise if no fallback
```

### Validate Parameters Early

**Good**:
```python
def validate_params(params, required):
    missing = [p for p in required if p not in params]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")

async def render_safely(builder, prompt_name, params):
    # Validate early
    validate_params(params, ["code", "language"])

    # Then render
    return await builder.render_user_prompt(prompt_name, params=params)
```

**Benefits**:
- Clear error messages
- Fail fast
- Easier debugging

### Handle LLM Errors with Retries

**Pattern**:
```python
import asyncio

async def complete_with_retry(manager, max_retries=3, backoff=1.0):
    for attempt in range(max_retries):
        try:
            return await manager.complete()
        except Exception as e:
            if attempt == max_retries - 1:
                # Final attempt failed
                logger.error(f"LLM failed after {max_retries} retries: {e}")
                raise

            # Exponential backoff
            wait = backoff * (2 ** attempt)
            logger.warning(f"LLM attempt {attempt+1} failed, retrying in {wait}s")
            await asyncio.sleep(wait)
```

### Log Errors with Context

**Good**:
```python
try:
    result = await manager.complete()
except Exception as e:
    logger.error(
        "LLM completion failed",
        extra={
            "conversation_id": manager.conversation_id,
            "message_count": len(manager.state.get_current_messages()),
            "error": str(e),
            "user_id": manager.state.metadata.get("user_id")
        }
    )
    raise
```

**Benefits**:
- Easier debugging
- Better monitoring
- Context for error analysis

---

## Testing

### Test Prompts in Isolation

**Pattern**:
```python
import pytest
from dataknobs_llm.prompts import render_template

def test_code_review_prompt():
    template = "Review {{language}} code:\n{{code}}"

    result = render_template(
        template,
        {"language": "Python", "code": "def test(): pass"}
    )

    assert "Review Python code:" in result.content
    assert "def test(): pass" in result.content

def test_optional_params():
    template = "Hello {{name}}((, age {{age}}))"

    # With optional param
    result = render_template(template, {"name": "Alice", "age": 30})
    assert "age 30" in result.content

    # Without optional param
    result = render_template(template, {"name": "Alice"})
    assert "age" not in result.content
```

### Use EchoProvider for Testing

**Pattern**:
```python
from dataknobs_llm.llm import EchoProvider

@pytest.mark.asyncio
async def test_conversation_flow():
    # Echo provider returns input for predictable testing
    llm = EchoProvider(config={"echo_prefix": "Echo: "})

    manager = await ConversationManager.create(
        llm=llm,
        prompt_builder=builder,
        storage=storage
    )

    await manager.add_message(role="user", content="Hello")
    response = await manager.complete()

    # Predictable response
    assert "Echo: " in response.content
    assert "Hello" in response.content
```

### Test Middleware Behavior

**Pattern**:
```python
@pytest.mark.asyncio
async def test_logging_middleware(caplog):
    logger = logging.getLogger("test")
    middleware = LoggingMiddleware(logger)

    manager = await ConversationManager.create(
        llm=echo_llm,
        prompt_builder=builder,
        storage=storage,
        middleware=[middleware]
    )

    with caplog.at_level(logging.INFO):
        await manager.add_message(role="user", content="Test")
        await manager.complete()

    # Check logs
    assert "Sending" in caplog.text
    assert "Received response" in caplog.text
```

### Test Error Conditions

**Pattern**:
```python
@pytest.mark.asyncio
async def test_missing_required_param():
    from dataknobs_llm.prompts import ValidationLevel

    with pytest.raises(ValueError, match="Missing required parameter"):
        result = await builder.render_user_prompt(
            "prompt_with_required_param",
            params={},  # Missing required param
            validation_level=ValidationLevel.ERROR
        )
```

### Mock External Dependencies

**Pattern**:
```python
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_with_mocked_adapter():
    # Mock adapter
    mock_adapter = AsyncMock()
    mock_adapter.search.return_value = [
        {"content": "Test result", "score": 0.9}
    ]

    builder = AsyncPromptBuilder(
        library=library,
        adapters={"docs": mock_adapter}
    )

    result = await builder.render_system_prompt(
        "prompt_with_rag",
        params={"topic": "testing"}
    )

    # Verify adapter was called
    mock_adapter.search.assert_called_once()
```

---

## Production Deployment

### Use Environment-Specific Configurations

**Pattern**:
```python
import os

def get_storage(env=None):
    env = env or os.getenv("ENV", "development")

    if env == "production":
        return DataknobsConversationStorage(
            AsyncPostgresDatabase(
                connection_string=os.getenv("DATABASE_URL")
            )
        )
    elif env == "staging":
        return DataknobsConversationStorage(
            AsyncPostgresDatabase(
                connection_string=os.getenv("STAGING_DATABASE_URL")
            )
        )
    else:  # development
        return DataknobsConversationStorage(
            AsyncSQLiteDatabase(db_path="dev_conversations.db")
        )
```

### Implement Health Checks

**Pattern**:
```python
async def health_check():
    """Check if conversation system is healthy."""
    try:
        # Test storage
        test_state = ConversationState(
            conversation_id="health-check",
            message_tree=Tree(ConversationNode(...))
        )
        await storage.save_conversation(test_state)
        loaded = await storage.load_conversation("health-check")
        await storage.delete_conversation("health-check")

        if not loaded:
            return {"status": "unhealthy", "reason": "storage failed"}

        # Test LLM
        response = await llm.complete([
            LLMMessage(role="user", content="health check")
        ])

        if not response:
            return {"status": "unhealthy", "reason": "llm failed"}

        return {"status": "healthy"}

    except Exception as e:
        return {"status": "unhealthy", "reason": str(e)}
```

### Monitor Token Usage

**Pattern**:
```python
class TokenMonitoringMiddleware(ConversationMiddleware):
    def __init__(self, alert_threshold=50000):
        self.alert_threshold = alert_threshold
        self.total_tokens = 0

    async def process_response(self, response, state):
        if response.usage:
            tokens = response.usage.get("total_tokens", 0)
            self.total_tokens += tokens

            # Log to metrics system
            metrics.increment("llm.tokens", tokens)

            # Alert if high usage
            if tokens > self.alert_threshold:
                logger.warning(f"High token usage: {tokens}")

            response.metadata["total_tokens"] = tokens

        return response
```

### Implement Rate Limiting

**Pattern**:
```python
from collections import defaultdict
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, max_requests=100, window_minutes=1):
        self.max_requests = max_requests
        self.window = timedelta(minutes=window_minutes)
        self.requests = defaultdict(list)  # user_id -> [timestamps]

    def is_allowed(self, user_id):
        now = datetime.now()
        cutoff = now - self.window

        # Clean old requests
        self.requests[user_id] = [
            ts for ts in self.requests[user_id]
            if ts > cutoff
        ]

        # Check limit
        if len(self.requests[user_id]) >= self.max_requests:
            return False

        # Record request
        self.requests[user_id].append(now)
        return True

# Usage
rate_limiter = RateLimiter(max_requests=10, window_minutes=1)

async def handle_request(user_id, message):
    if not rate_limiter.is_allowed(user_id):
        raise ValueError("Rate limit exceeded")

    # Process request...
```

### Secure API Keys

**Good**:
```python
import os

# Load from environment
llm = OpenAIProvider(config={
    "api_key": os.getenv("OPENAI_API_KEY")
})
```

**Avoid**:
```python
# Hardcoded keys - NEVER DO THIS
llm = OpenAIProvider(config={
    "api_key": "sk-hardcoded-key-12345"
})
```

### Implement Graceful Shutdown

**Pattern**:
```python
import signal
import asyncio

class ConversationService:
    def __init__(self):
        self.shutdown_event = asyncio.Event()

    async def run(self):
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Run until shutdown
        await self.shutdown_event.wait()

        # Cleanup
        await self.cleanup()

    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received")
        self.shutdown_event.set()

    async def cleanup(self):
        # Close LLM connections
        await self.llm.close()

        # Flush any pending saves
        # ...

        logger.info("Shutdown complete")
```

---

## Summary

### Quick Reference

**Prompt Design**:
- Clear, focused prompts
- Sensible defaults
- Limited conditional nesting (max 2 levels)
- Use template inheritance

**Validation**:
- ERROR for critical params
- WARN for recommended params
- Document all parameters

**RAG**:
- Separate configs from prompts
- Parameterize queries
- Handle failures gracefully
- Start with k=5, threshold=0.7

**Conversations**:
- Meaningful metadata
- Implement limits
- Clean up old conversations
- Use branching for exploration

**Middleware**:
- Order thoughtfully (log first, metadata last)
- Use validation selectively (cost consideration)
- Handle errors appropriately
- Monitor performance

**Storage**:
- Choose backend for your scale
- Implement caching for high traffic
- Batch operations
- Monitor performance

**Production**:
- Environment-specific configs
- Health checks
- Rate limiting
- Monitor token usage
- Secure credentials
- Graceful shutdown

---

**Follow these practices for robust, maintainable, and efficient LLM applications!**

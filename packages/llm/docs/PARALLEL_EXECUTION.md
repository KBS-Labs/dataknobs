# Parallel LLM Execution

The `ParallelLLMExecutor` runs multiple LLM calls (and deterministic functions) concurrently with concurrency control, per-task error isolation, and optional per-task retry.

## Overview

```python
from dataknobs_llm import ParallelLLMExecutor, LLMTask, LLMMessage

executor = ParallelLLMExecutor(provider, max_concurrency=3)

results = await executor.execute({
    "q1": LLMTask(messages=[LLMMessage(role="user", content="Generate a math question")]),
    "q2": LLMTask(messages=[LLMMessage(role="user", content="Generate a science question")]),
    "q3": LLMTask(messages=[LLMMessage(role="user", content="Generate a history question")]),
})

for tag, result in results.items():
    if result.success:
        print(f"{tag}: {result.value.content}")
    else:
        print(f"{tag}: FAILED - {result.error}")
```

## ParallelLLMExecutor

### Creating an Executor

```python
from dataknobs_llm import ParallelLLMExecutor
from dataknobs_common.retry import RetryConfig, BackoffStrategy

executor = ParallelLLMExecutor(
    provider=llm_provider,        # Any AsyncLLMProvider
    max_concurrency=5,            # Semaphore limit
    default_retry=RetryConfig(    # Optional default retry policy
        max_attempts=3,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
    ),
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `AsyncLLMProvider` | required | The LLM provider for executing tasks |
| `max_concurrency` | `int` | `5` | Maximum concurrent tasks (semaphore limit) |
| `default_retry` | `RetryConfig \| None` | `None` | Default retry policy for tasks without their own |

### execute()

Runs LLM tasks concurrently with error isolation:

```python
results = await executor.execute({
    "stem": LLMTask(messages=[LLMMessage(role="user", content="Generate a question stem")]),
    "distractors": LLMTask(messages=[LLMMessage(role="user", content="Generate distractors")]),
})
```

- **Input**: `dict[str, LLMTask]` — mapping of tag to task
- **Output**: `dict[str, TaskResult]` — mapping of tag to result
- Each task runs independently; one failure does not cancel others
- Concurrency is controlled by `max_concurrency`

### execute_mixed()

Runs a mix of LLM and deterministic tasks concurrently:

```python
from dataknobs_llm import DeterministicTask

results = await executor.execute_mixed({
    "question": LLMTask(
        messages=[LLMMessage(role="user", content="Generate a question")],
    ),
    "timestamp": DeterministicTask(
        fn=lambda: datetime.now().isoformat(),
    ),
    "lookup": DeterministicTask(
        fn=fetch_reference_data,
        args=("biology",),
    ),
})
```

Deterministic tasks:
- Sync callables are run in a thread executor to avoid blocking the event loop
- Async callables are awaited directly
- Share the same concurrency semaphore as LLM tasks

### execute_sequential()

Runs LLM tasks in order, optionally passing results forward:

```python
results = await executor.execute_sequential(
    tasks=[
        LLMTask(
            messages=[LLMMessage(role="user", content="Generate a question")],
            tag="generate",
        ),
        LLMTask(
            messages=[LLMMessage(role="user", content="Now improve this question")],
            tag="improve",
        ),
    ],
    pass_result=True,  # Appends previous response as assistant message
)
```

When `pass_result=True`, each task's messages are augmented with the previous task's response as an assistant message, creating a chain where each step builds on the last.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tasks` | `list[LLMTask]` | required | Ordered tasks to run |
| `pass_result` | `bool` | `False` | Append previous result as assistant message |

Returns `list[TaskResult]` in execution order.

## LLMTask

Represents a single LLM call:

```python
from dataknobs_llm import LLMTask, LLMMessage
from dataknobs_common.retry import RetryConfig

task = LLMTask(
    messages=[
        LLMMessage(role="system", content="You are a quiz generator."),
        LLMMessage(role="user", content="Generate a biology question."),
    ],
    config_overrides={"temperature": 0.9},  # Per-task provider overrides
    retry=RetryConfig(max_attempts=2),       # Per-task retry policy
    tag="bio_q1",                            # Auto-set from dict key in execute()
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `messages` | `list[LLMMessage]` | required | Messages to send to the provider |
| `config_overrides` | `dict[str, Any] \| None` | `None` | Per-task config (temperature, model, etc.) |
| `retry` | `RetryConfig \| None` | `None` | Per-task retry (overrides executor default) |
| `tag` | `str` | `""` | Identifier; auto-populated from dict key |

## DeterministicTask

Represents a sync or async callable:

```python
from dataknobs_llm import DeterministicTask

# Sync function
task = DeterministicTask(
    fn=compute_hash,
    args=("content",),
    kwargs={"algorithm": "sha256"},
)

# Async function
task = DeterministicTask(
    fn=fetch_from_database,
    args=("record_123",),
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fn` | `Callable[..., Any]` | required | The callable (sync or async) |
| `args` | `tuple[Any, ...]` | `()` | Positional arguments |
| `kwargs` | `dict[str, Any]` | `{}` | Keyword arguments |
| `tag` | `str` | `""` | Identifier; auto-populated from dict key |

## TaskResult

Result of a single task execution:

```python
result = results["q1"]

if result.success:
    response = result.value  # LLMResponse for LLM tasks, Any for deterministic
    print(f"Completed in {result.duration_ms:.1f}ms")
else:
    print(f"Failed: {result.error}")
```

| Field | Type | Description |
|-------|------|-------------|
| `tag` | `str` | Task identifier |
| `success` | `bool` | Whether the task completed without error |
| `value` | `LLMResponse \| Any` | Return value (`None` on failure) |
| `error` | `Exception \| None` | The exception if failed |
| `duration_ms` | `float` | Wall-clock execution time in milliseconds |

## Error Handling

Each task is isolated — a failure in one task does not affect others:

```python
results = await executor.execute({
    "good": LLMTask(messages=[LLMMessage(role="user", content="Hello")]),
    "bad": LLMTask(messages=[LLMMessage(role="user", content="trigger error")]),
})

assert results["good"].success is True   # Unaffected by other failures
assert results["bad"].success is False
assert results["bad"].error is not None
```

### Retry

Tasks can specify retry policies individually or inherit the executor's default:

```python
from dataknobs_common.retry import RetryConfig, BackoffStrategy

# Executor-level default
executor = ParallelLLMExecutor(
    provider=provider,
    default_retry=RetryConfig(max_attempts=3),
)

# Per-task override
results = await executor.execute({
    "critical": LLMTask(
        messages=[LLMMessage(role="user", content="Important task")],
        retry=RetryConfig(
            max_attempts=5,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            initial_delay=1.0,
        ),
    ),
    "best_effort": LLMTask(
        messages=[LLMMessage(role="user", content="Optional task")],
        retry=RetryConfig(max_attempts=1),  # No retry
    ),
})
```

Retry uses `RetryExecutor` from `dataknobs-common`, which supports fixed, linear, exponential, jitter, and decorrelated backoff strategies.

## Testing

Use `EchoProvider` for deterministic testing:

```python
from dataknobs_llm import EchoProvider, ParallelLLMExecutor, LLMTask, LLMMessage
from dataknobs_llm.testing import text_response

provider = EchoProvider()
provider.set_responses([
    text_response("Math question"),
    text_response("Science question"),
])

executor = ParallelLLMExecutor(provider, max_concurrency=2)
results = await executor.execute({
    "math": LLMTask(messages=[LLMMessage(role="user", content="math")]),
    "science": LLMTask(messages=[LLMMessage(role="user", content="science")]),
})

assert results["math"].success
assert results["science"].success
```

### Testing Error Handling

`EchoProvider` supports `ErrorResponse` for simulating failures:

```python
from dataknobs_llm.testing import text_response, ErrorResponse

provider = EchoProvider()
provider.set_responses([
    text_response("OK"),
    ErrorResponse(RuntimeError("simulated failure")),
])
```

## Imports

All main types are available from the top-level `dataknobs_llm` package:

```python
from dataknobs_llm import (
    ParallelLLMExecutor,
    LLMTask,
    DeterministicTask,
    TaskResult,
    LLMMessage,
)
```

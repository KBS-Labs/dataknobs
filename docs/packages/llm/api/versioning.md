# Versioning & A/B Testing API

Prompt version management, A/B testing, and metrics tracking.

> **📖 Also see:** [Auto-generated API Reference](../../../api/reference/llm.md) - Complete documentation from source code docstrings

---

## Overview

The versioning API provides comprehensive tools for tracking prompt versions, running A/B tests, and measuring performance.

## Version Management

### VersionManager

::: dataknobs_llm.prompts.VersionManager
    options:
      show_source: true
      heading_level: 3
      members:
        - create_version
        - get_version
        - list_versions
        - list_names
        - update_status
        - tag_version
        - untag_version
        - get_version_history

### PromptVersion

::: dataknobs_llm.prompts.PromptVersion
    options:
      show_source: true
      heading_level: 3

### VersionStatus

::: dataknobs_llm.prompts.VersionStatus
    options:
      show_source: true
      heading_level: 3

## A/B Testing

### ABTestManager

::: dataknobs_llm.prompts.ABTestManager
    options:
      show_source: true
      heading_level: 3
      members:
        - create_experiment
        - get_experiment
        - list_experiments
        - get_variant_for_user
        - get_random_variant
        - update_experiment_status
        - get_variant_distribution

### PromptExperiment

::: dataknobs_llm.prompts.PromptExperiment
    options:
      show_source: true
      heading_level: 3

### PromptVariant

::: dataknobs_llm.prompts.PromptVariant
    options:
      show_source: true
      heading_level: 3

## Metrics Tracking

### MetricsCollector

::: dataknobs_llm.prompts.MetricsCollector
    options:
      show_source: true
      heading_level: 3
      members:
        - record_event
        - get_metrics
        - compare_variants
        - get_top_versions

### PromptMetrics

::: dataknobs_llm.prompts.PromptMetrics
    options:
      show_source: true
      heading_level: 3

### MetricEvent

::: dataknobs_llm.prompts.MetricEvent
    options:
      show_source: true
      heading_level: 3

## Storage

Where the three managers keep what they are given. The default is in memory;
`DatabaseVersionStore` puts everything into any dataknobs `AsyncDatabase`.

Each manager declares only the part of the surface it uses, so a store written
for one need not implement the others. `VersioningStore` is all three at once,
which is what `VersionedPromptLibrary` asks for.

### VersionStore

::: dataknobs_llm.prompts.VersionStore
    options:
      show_source: false
      heading_level: 3

### ExperimentStore

::: dataknobs_llm.prompts.ExperimentStore
    options:
      show_source: false
      heading_level: 3

### MetricsStore

::: dataknobs_llm.prompts.MetricsStore
    options:
      show_source: false
      heading_level: 3

### VersioningStore

::: dataknobs_llm.prompts.VersioningStore
    options:
      show_source: false
      heading_level: 3

### InMemoryVersionStore

::: dataknobs_llm.prompts.InMemoryVersionStore
    options:
      show_source: false
      heading_level: 3

### DatabaseVersionStore

::: dataknobs_llm.prompts.DatabaseVersionStore
    options:
      show_source: false
      heading_level: 3

## Usage Examples

### Persisting to a Backend

```python
from dataknobs_data import async_database_factory
from dataknobs_llm.prompts import (
    ABTestManager,
    DatabaseVersionStore,
    MetricsCollector,
    VersionManager,
)

# "memory", "file", "sqlite", "postgres", "s3", "duckdb", "elasticsearch".
# The factory builds the database; it does not connect it.
db = async_database_factory.create(backend="sqlite", path="./prompts.db")
await db.connect()

# One store for all three managers, so they share a database rather than
# three configurations of one.
store = DatabaseVersionStore(db)
vm = VersionManager(store)
ab = ABTestManager(store)
mc = MetricsCollector(store)

# ... use them ...

# The store does not own the database. You opened it; you close it.
await db.close()
```

A load hands back a value rather than a handle, so an in-place change is not
stored until it is saved:

```python
version = await store.load_version(version_id)
version.tags.append("production")
await store.save_version(version)
```

That holds for `InMemoryVersionStore` too, so code developed against it
behaves the same against a database.

### Version Management

```python
from dataknobs_llm.prompts import VersionManager, VersionStatus

# Create version manager
vm = VersionManager()

# Create first version
v1 = await vm.create_version(
    name="greeting",
    prompt_type="system",
    template="Hello {{name}}!",
    version="1.0.0",
    metadata={"author": "alice", "description": "Initial version"}
)

# Auto-increment version
v2 = await vm.create_version(
    name="greeting",
    prompt_type="system",
    template="Hi {{name}}, welcome!",
    # version="1.0.1" automatically assigned
    metadata={"author": "bob", "changes": "More friendly"}
)

# Get specific version
version = await vm.get_version("greeting", "system", "1.0.0")

# Get latest version (no version specified)
latest = await vm.get_version("greeting", "system")

# List all versions
versions = await vm.list_versions("greeting", "system")
for v in versions:
    print(f"{v.version}: {v.metadata.get('description')}")
```

### Version Status Lifecycle

```python
# Create as draft
draft = await vm.create_version(
    name="new_feature",
    prompt_type="user",
    template="...",
    status=VersionStatus.DRAFT
)

# Promote to active
await vm.update_status(draft.version_id, VersionStatus.ACTIVE)

# Tag as production
await vm.tag_version(draft.version_id, "production")

# Later, deprecate old version
await vm.update_status(old_version.version_id, VersionStatus.DEPRECATED)
```

### A/B Testing

```python
from dataknobs_llm.prompts import ABTestManager, PromptVariant

# Create A/B test manager
ab = ABTestManager()

# Create experiment
exp = await ab.create_experiment(
    name="greeting",
    prompt_type="system",
    variants=[
        PromptVariant("1.0.0", 0.5, "Control"),
        PromptVariant("1.0.1", 0.5, "Treatment")
    ],
    metadata={
        "hypothesis": "Friendly greeting increases engagement",
        "owner": "product-team"
    }
)

# Get variant for user (sticky - same user always gets same variant)
variant = await ab.get_variant_for_user(exp.experiment_id, "user123")
print(f"User 123 gets version: {variant}")

# Same user always gets same variant
variant2 = await ab.get_variant_for_user(exp.experiment_id, "user123")
assert variant == variant2

# Random variant (for batch processing)
random_variant = await ab.get_random_variant(exp.experiment_id)
```

### Multi-Variant Testing (A/B/C)

```python
# Create A/B/C test
exp = await ab.create_experiment(
    name="tone",
    prompt_type="system",
    variants=[
        PromptVariant("1.0.0", 0.34, "Formal"),
        PromptVariant("1.1.0", 0.33, "Casual"),
        PromptVariant("1.2.0", 0.33, "Enthusiastic")
    ]
)

# Traffic automatically normalized to sum to 1.0
```

### Unequal Traffic Splits

```python
# Safer rollout: 90% control, 10% experimental
exp = await ab.create_experiment(
    name="new_feature",
    prompt_type="system",
    variants=[
        PromptVariant("1.0.0", 0.9, "Stable"),
        PromptVariant("2.0.0", 0.1, "Experimental")
    ]
)
```

### Metrics Tracking

```python
from dataknobs_llm.prompts import MetricsCollector

# Create metrics collector
mc = MetricsCollector()

# Record comprehensive event
await mc.record_event(
    version_id=version.version_id,
    success=True,
    response_time=0.5,  # seconds
    tokens=150,
    user_rating=4.5,  # 1-5 scale
    metadata={"user_segment": "premium", "feature": "code_review"}
)

# Record minimal event
await mc.record_event(
    version_id=version.version_id,
    success=True
)

# Record failure
await mc.record_event(
    version_id=version.version_id,
    success=False,
    metadata={"error": "timeout"}
)
```

### Analyzing Results

```python
# Get metrics for a version
metrics = await mc.get_metrics(version.version_id)
print(f"Success rate: {metrics.success_rate:.2%}")
print(f"Total events: {metrics.total_events}")
print(f"Avg response time: {metrics.avg_response_time:.2f}s")
print(f"Avg tokens: {metrics.avg_tokens:.0f}")
print(f"Avg rating: {metrics.avg_rating:.1f}/5.0")

# Compare variants
comparison = await mc.compare_variants([v1.version_id, v2.version_id])

for version_id, metrics in comparison.items():
    print(f"\nVersion {version_id}:")
    print(f"  Success rate: {metrics.success_rate:.2%}")
    print(f"  Avg response time: {metrics.avg_response_time:.2f}s")
    print(f"  Avg rating: {metrics.avg_rating:.1f}/5.0")
    print(f"  Total events: {metrics.total_events}")

# Find top performers
top_versions = await mc.get_top_versions(
    version_ids=[v1.version_id, v2.version_id, v3.version_id],
    metric="success_rate",
    limit=3
)

print("Top 3 versions by success rate:")
for version_id, metrics in top_versions:
    print(f"  {version_id}: {metrics.success_rate:.2%}")
```

### Full A/B Test Workflow

```python
from dataknobs_llm.prompts import (
    VersionManager,
    ABTestManager,
    MetricsCollector,
    PromptVariant
)

# 1. Create versions
vm = VersionManager()
v1 = await vm.create_version(
    name="email_subject",
    prompt_type="user",
    template="Generate subject for: {{topic}}",
    version="1.0.0"
)

v2 = await vm.create_version(
    name="email_subject",
    prompt_type="user",
    template="Generate engaging subject line for: {{topic}}",
    version="1.1.0"
)

# 2. Create experiment
ab = ABTestManager()
exp = await ab.create_experiment(
    name="email_subject",
    prompt_type="user",
    variants=[
        PromptVariant("1.0.0", 0.5, "Basic"),
        PromptVariant("1.1.0", 0.5, "Engaging")
    ]
)

# 3. Run experiment
mc = MetricsCollector()

for user_id in users:
    # Get variant for user
    variant_version = await ab.get_variant_for_user(exp.experiment_id, user_id)
    version = await vm.get_version("email_subject", "user", variant_version)

    # Use version in your application
    success = execute_with_version(version, user_id)

    # Track metrics
    await mc.record_event(
        version_id=version.version_id,
        success=success,
        response_time=response_time,
        user_rating=user_rating
    )

# 4. Analyze and deploy winner
comparison = await mc.compare_variants([v1.version_id, v2.version_id])

if comparison[v2.version_id].success_rate > comparison[v1.version_id].success_rate:
    # Deploy winner
    await vm.tag_version(v2.version_id, "production")
    await ab.update_experiment_status(exp.experiment_id, "completed")
else:
    # Keep control
    await vm.tag_version(v1.version_id, "production")
```

### Rollback Scenario

```python
# Deploy new version
new_version = await vm.create_version(
    name="greeting",
    prompt_type="system",
    template="New experimental greeting",
    version="2.0.0"
)
await vm.tag_version(new_version.version_id, "production")

# Monitor metrics...
metrics = await mc.get_metrics(new_version.version_id)

if metrics.success_rate < 0.9:
    # Rollback!
    versions = await vm.list_versions("greeting", "system")
    previous = versions[1]  # Second newest

    await vm.untag_version(new_version.version_id, "production")
    await vm.tag_version(previous.version_id, "production")
    await vm.update_status(new_version.version_id, VersionStatus.DEPRECATED)

    print(f"Rolled back to version {previous.version}")
```

### Versioned Prompt Library

```python
from dataknobs_llm.prompts import VersionedPromptLibrary

# Create versioned library
library = VersionedPromptLibrary()

# Create versions
await library.create_version(
    name="greeting",
    prompt_type="system",
    template="Hello {{name}}!",
    version="1.0.0"
)

# Create A/B test
await library.create_experiment(
    name="greeting",
    prompt_type="system",
    variants=[
        PromptVariant("1.0.0", 0.5, "Control"),
        PromptVariant("1.0.1", 0.5, "Treatment")
    ]
)

# Pick the variant for a user, then fetch that version. Selection is explicit
# -- the prompt builder has no user_id and does not choose for you.
experiment = await library.get_experiment(experiment_id)
version = await library.get_variant_for_user(experiment.experiment_id, "user123")

# Sticky: the same user gets the same variant on every later turn.
template = await library.get_version("greeting", "system", version)
```

!!! note "This library is the asynchronous flavour"

    `VersionedPromptLibrary` implements `AsyncPromptLibrary`, not
    `AbstractPromptLibrary`: every answer it gives comes from a version
    manager that awaits, so its accessors are coroutines like its writers.

    ```python
    template = await library.get_system_prompt("greeting")
    names = await library.list_system_prompts()
    ```

    A `def` caller reaches it through `as_sync`, which runs the coroutine on a
    private event loop — so the call returns rather than raising, at the cost
    of a daemon thread and a blocked calling thread for the whole call:

    ```python
    from dataknobs_llm.prompts import as_sync

    with as_sync(library, timeout=30) as view:
        template = view.get_system_prompt("greeting")
    ```

    The accessors were synchronous before, because the only library protocol
    available said so. They reached the async version manager with
    `loop.run_until_complete`, which raises `RuntimeError: This event loop is
    already running` for a caller already on one — and a running loop is the
    only place this library can be populated from, because `create_version`
    and `create_experiment` are coroutines.

## Best Practices

### 1. Semantic Versioning

```python
# Major: Breaking changes
await vm.create_version(..., version="2.0.0")

# Minor: New features, backward compatible
await vm.create_version(..., version="1.1.0")

# Patch: Bug fixes, minor improvements
await vm.create_version(..., version="1.0.1")
```

### 2. Comprehensive Metadata

```python
await vm.create_version(
    ...,
    metadata={
        "author": "alice",
        "description": "Fixed hallucination issue",
        "jira_ticket": "PROMPT-123",
        "tested": True,
        "review_approved_by": "bob"
    }
)
```

### 3. User-Sticky for Consistency

```python
# ✅ User-facing apps
variant = await ab.get_variant_for_user(exp_id, user_id)

# ❌ Don't use random for user-facing (inconsistent UX)
variant = await ab.get_random_variant(exp_id)
```

### 4. Track All Relevant Metrics

```python
await mc.record_event(
    version_id=v_id,
    success=success,
    response_time=time,
    tokens=tokens,
    user_rating=rating,
    metadata={
        "context": "code_review",
        "language": "python",
        "user_segment": "premium"
    }
)
```

## See Also

- [Versioning & A/B Testing Guide](../guides/versioning.md) - Detailed guide
- [Prompts API](prompts.md) - Template system
- [Examples](../examples/ab-testing.md) - A/B testing workflows
- [Performance Guide](../guides/performance.md) - Optimization tips

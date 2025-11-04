# Versioning & A/B Testing

Track prompt versions, run experiments, and measure performance.

## Overview

The versioning system provides:

- **Semantic Versioning**: Track prompts with major.minor.patch
- **A/B Testing**: Multi-variant experiments with traffic splitting
- **User-Sticky Assignment**: Deterministic variant assignment
- **Metrics Tracking**: Success rates, response times, ratings
- **Performance Comparison**: Compare variants to find winners
- **Rollback**: Easy rollback to previous versions

## Quick Start

```python
from dataknobs_llm.prompts import (
    VersionManager,
    ABTestManager,
    MetricsCollector,
    PromptVariant
)

# Create managers
vm = VersionManager()
ab = ABTestManager()
mc = MetricsCollector()

# Create versions
v1 = await vm.create_version(
    name="greeting",
    prompt_type="system",
    template="Hello {{name}}!",
    version="1.0.0"
)

v2 = await vm.create_version(
    name="greeting",
    prompt_type="system",
    template="Hi {{name}}, welcome!"  # Auto: v1.0.1
)

# Create A/B test
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

# Track metrics
await mc.record_event(
    version_id=v2.version_id,
    success=True,
    response_time=0.5,
    tokens=100,
    user_rating=4.5
)

# Compare performance
comparison = await mc.compare_variants([v1.version_id, v2.version_id])
for version_id, metrics in comparison.items():
    print(f"{version_id}: {metrics.success_rate:.1%} success")
```

## Version Management

### Creating Versions

```python
# With explicit version
v1 = await vm.create_version(
    name="code_analysis",
    prompt_type="system",
    template="Analyze this {{language}} code: {{code}}",
    version="1.0.0",
    metadata={"author": "alice"}
)

# Auto-increment
v2 = await vm.create_version(
    name="code_analysis",
    prompt_type="system",
    template="Analyze this {{language}} code and suggest improvements: {{code}}",
    # version="1.0.1" automatically assigned
    metadata={"author": "bob", "changes": "added suggestions"}
)
```

### Version Status

```python
from dataknobs_llm.prompts import VersionStatus

# Create as draft
draft = await vm.create_version(
    name="new_feature",
    prompt_type="user",
    template="...",
    status=VersionStatus.DRAFT
)

# Promote to production
await vm.update_status(draft.version_id, VersionStatus.PRODUCTION)

# Tag version
await vm.tag_version(draft.version_id, "production")
```

## A/B Testing

### Creating Experiments

```python
# 50/50 split
experiment = await ab.create_experiment(
    name="greeting",
    prompt_type="system",
    variants=[
        PromptVariant("1.0.0", 0.5, "Control"),
        PromptVariant("1.1.0", 0.5, "Treatment")
    ],
    metadata={
        "hypothesis": "Friendly greeting increases engagement",
        "owner": "product-team"
    }
)

# Unequal split (safer for risky changes)
experiment = await ab.create_experiment(
    name="new_feature",
    prompt_type="system",
    variants=[
        PromptVariant("1.0.0", 0.9, "Stable"),
        PromptVariant("2.0.0", 0.1, "Experimental")
    ]
)

# Multi-variant (A/B/C)
experiment = await ab.create_experiment(
    name="tone",
    prompt_type="system",
    variants=[
        PromptVariant("1.0.0", 0.34, "Formal"),
        PromptVariant("1.1.0", 0.33, "Casual"),
        PromptVariant("1.2.0", 0.33, "Enthusiastic")
    ]
)
```

### Variant Selection

```python
# User-sticky (recommended for user-facing)
variant = await ab.get_variant_for_user(
    experiment.experiment_id,
    user_id="user123"
)
# Same user always gets same variant

# Random (for batch processing)
variant = await ab.get_random_variant(experiment.experiment_id)
```

## Metrics Tracking

### Recording Events

```python
# Comprehensive event
await mc.record_event(
    version_id=version.version_id,
    success=True,
    response_time=0.5,  # seconds
    tokens=150,
    user_rating=4.5,  # 1-5 scale
    metadata={"user_segment": "premium"}
)

# Minimal event
await mc.record_event(version_id=version.version_id, success=True)
```

### Analyzing Results

```python
# Get metrics for a version
metrics = await mc.get_metrics(version_id)
print(f"Success rate: {metrics.success_rate:.2%}")
print(f"Avg response time: {metrics.avg_response_time:.2f}s")
print(f"Avg rating: {metrics.avg_rating:.1f}/5.0")

# Compare all variants
comparison = await mc.compare_variants([v1.version_id, v2.version_id])

# Find top performers
top = await mc.get_top_versions(
    [v1.version_id, v2.version_id, v3.version_id],
    metric="success_rate",
    limit=3
)
```

## Detailed Documentation

Comprehensive versioning and A/B testing documentation:

### Versioning Guide

**Location**: `packages/llm/docs/VERSIONING.md`

Topics covered:
- Semantic versioning
- Version status lifecycle
- Tagging system
- Parent tracking
- Version history
- Rollback procedures
- Best practices

### A/B Testing Guide

**Location**: `packages/llm/docs/AB_TESTING.md`

Topics covered:
- Creating experiments
- Variant selection strategies (sticky vs random)
- Traffic split management
- Metrics tracking
- Performance comparison
- Statistical significance
- Experiment lifecycle
- Best practices

## Common Patterns

### Full A/B Test Workflow

```python
# 1. Create versions
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
exp = await ab.create_experiment(
    name="email_subject",
    prompt_type="user",
    variants=[
        PromptVariant("1.0.0", 0.5, "Basic"),
        PromptVariant("1.1.0", 0.5, "Engaging")
    ]
)

# 3. Run for users
for user_id in users:
    variant = await ab.get_variant_for_user(exp.experiment_id, user_id)
    version = await vm.get_version("email_subject", "user", variant)

    # Use version...
    success = execute_with_version(version)

    # Track metrics
    await mc.record_event(version.version_id, success=success)

# 4. Analyze and deploy winner
comparison = await mc.compare_variants([v1.version_id, v2.version_id])
if comparison[v2.version_id].success_rate > comparison[v1.version_id].success_rate:
    await vm.tag_version(v2.version_id, "production")
    await ab.update_experiment_status(exp.experiment_id, "completed")
```

### Rollback Scenario

```python
# Deploy new version
new_version = await vm.create_version(
    name="greeting",
    prompt_type="system",
    template="New version",
    version="2.0.0"
)
await vm.tag_version(new_version.version_id, "production")

# Monitor...detect issues...

# Rollback
versions = await vm.list_versions("greeting", "system")
previous = versions[1]  # Second newest

await vm.untag_version(new_version.version_id, "production")
await vm.tag_version(previous.version_id, "production")
await vm.update_status(new_version.version_id, VersionStatus.DEPRECATED)
```

## Best Practices

1. **Use Semantic Versioning Correctly**
   - Major: Breaking changes
   - Minor: New features, backward compatible
   - Patch: Bug fixes, minor improvements

2. **Always Add Metadata**
   ```python
   await vm.create_version(
       ...,
       metadata={
           "author": "alice",
           "description": "Fixed hallucination issue",
           "jira_ticket": "PROMPT-123",
           "tested": True
       }
   )
   ```

3. **Use User-Sticky for Consistency**
   ```python
   # ✅ User-facing apps
   variant = await ab.get_variant_for_user(exp_id, user_id)

   # ❌ Don't use random for user-facing
   variant = await ab.get_random_variant(exp_id)  # Inconsistent UX
   ```

4. **Track Comprehensive Metrics**
   ```python
   await mc.record_event(
       version_id=v_id,
       success=success,
       response_time=time,
       tokens=tokens,
       user_rating=rating,
       metadata={"context": "important"}
   )
   ```

5. **Run Experiments Long Enough**
   - Minimum 7 days for user-facing features
   - Ensure statistical significance
   - Consider weekly patterns

## See Also

- [Prompt Engineering](prompts.md) - Building prompts
- [Performance](performance.md) - Benchmarking and optimization
- [API Reference](../api/versioning.md) - Complete API
- [Examples](../examples/ab-testing.md) - Example workflows

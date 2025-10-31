# A/B Testing & Metrics Guide

This guide covers A/B testing and metrics tracking for prompts in dataknobs-llm.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Creating Experiments](#creating-experiments)
- [Variant Selection Strategies](#variant-selection-strategies)
- [Metrics Tracking](#metrics-tracking)
- [Best Practices](#best-practices)
- [Examples](#examples)

## Overview

The A/B testing system provides:

- **Experiment Management**: Create and manage multi-variant experiments
- **Traffic Splitting**: Control how traffic is distributed across variants
- **User-Sticky Assignment**: Consistent variant assignment per user
- **Random Selection**: Randomized variant selection
- **Metrics Tracking**: Monitor performance of each variant
- **Comparison Tools**: Compare variants to determine winners

## Quick Start

```python
from dataknobs_llm.prompts import (
    ABTestManager,
    PromptVariant,
    VersionManager,
    MetricsCollector
)

# Create managers
version_manager = VersionManager()
ab_manager = ABTestManager()
metrics = MetricsCollector()

# Create two versions to test
v1 = await version_manager.create_version(
    name="greeting",
    prompt_type="system",
    template="Hello {{name}}!",
    version="1.0.0"
)

v2 = await version_manager.create_version(
    name="greeting",
    prompt_type="system",
    template="Hi {{name}}, welcome!",
    version="1.1.0"
)

# Create A/B test experiment
experiment = await ab_manager.create_experiment(
    name="greeting",
    prompt_type="system",
    variants=[
        PromptVariant("1.0.0", 0.5, "Control - simple greeting"),
        PromptVariant("1.1.0", 0.5, "Treatment - welcoming greeting")
    ]
)

# Get variant for a user (sticky assignment)
variant_version = await ab_manager.get_variant_for_user(
    experiment.experiment_id,
    user_id="user123"
)

# Get the prompt version
version = await version_manager.get_version(
    name="greeting",
    prompt_type="system",
    version=variant_version
)

# Record metrics
await metrics.record_event(
    version_id=version.version_id,
    success=True,
    response_time=0.5,
    tokens=50,
    user_rating=4.5
)

# Compare variant performance
comparison = await metrics.compare_variants([v1.version_id, v2.version_id])
for version_id, metrics_data in comparison.items():
    print(f"Version: {version_id}")
    print(f"  Success rate: {metrics_data.success_rate:.2%}")
    print(f"  Avg rating: {metrics_data.avg_rating:.1f}/5.0")
```

## Creating Experiments

### Basic Experiment

```python
experiment = await ab_manager.create_experiment(
    name="code_analysis",
    prompt_type="system",
    variants=[
        PromptVariant("1.0.0", 0.5, "Basic analysis"),
        PromptVariant("1.1.0", 0.5, "Detailed analysis")
    ],
    metadata={
        "hypothesis": "Detailed analysis improves user satisfaction",
        "owner": "alice",
        "start_date": "2025-01-15"
    }
)
```

### Multi-Variant Testing (A/B/C)

```python
experiment = await ab_manager.create_experiment(
    name="greeting",
    prompt_type="system",
    variants=[
        PromptVariant("1.0.0", 0.34, "Formal"),
        PromptVariant("1.1.0", 0.33, "Casual"),
        PromptVariant("1.2.0", 0.33, "Enthusiastic")
    ]
)
```

### Unequal Traffic Split

```python
# 90% control, 10% treatment (safer for risky changes)
experiment = await ab_manager.create_experiment(
    name="new_feature",
    prompt_type="system",
    variants=[
        PromptVariant("1.0.0", 0.9, "Control - stable version"),
        PromptVariant("2.0.0", 0.1, "Treatment - new experimental feature")
    ]
)
```

### Custom Traffic Split

```python
# Use custom split instead of variant weights
experiment = await ab_manager.create_experiment(
    name="greeting",
    prompt_type="system",
    variants=[...],
    traffic_split={
        "1.0.0": 0.7,
        "1.1.0": 0.3
    }
)
```

## Variant Selection Strategies

### 1. User-Sticky Assignment (Recommended)

Same user always gets the same variant for consistent experience:

```python
# First request from user123
variant_v1 = await ab_manager.get_variant_for_user(
    experiment.experiment_id,
    user_id="user123"
)
# Returns: "1.0.0"

# Subsequent requests from user123
variant_v2 = await ab_manager.get_variant_for_user(
    experiment.experiment_id,
    user_id="user123"
)
# Returns: "1.0.0" (same as before)
```

**When to use**:
- User-facing applications
- When user experience consistency matters
- For tracking individual user journeys

**How it works**:
- Uses MD5 hash of user_id for deterministic assignment
- Same user_id always maps to same variant
- Distribution respects traffic split across all users

### 2. Random Selection

Each request gets a potentially different variant:

```python
# Each call may return different variant
variant1 = await ab_manager.get_random_variant(experiment.experiment_id)  # "1.0.0"
variant2 = await ab_manager.get_random_variant(experiment.experiment_id)  # "1.1.0"
variant3 = await ab_manager.get_random_variant(experiment.experiment_id)  # "1.0.0"
```

**When to use**:
- Non-user-facing applications
- Batch processing
- Testing traffic split distribution
- When user consistency doesn't matter

## Metrics Tracking

### Recording Events

```python
# Record successful usage
await metrics.record_event(
    version_id=version.version_id,
    success=True,
    response_time=0.5,  # seconds
    tokens=150,
    user_rating=4.5  # 1-5 scale
)

# Record failure
await metrics.record_event(
    version_id=version.version_id,
    success=False,
    metadata={"error": "timeout"}
)

# Minimal event
await metrics.record_event(
    version_id=version.version_id,
    success=True
)
```

### Viewing Metrics

```python
# Get metrics for a version
metrics_data = await metrics.get_metrics(version_id)

print(f"Total uses: {metrics_data.total_uses}")
print(f"Success rate: {metrics_data.success_rate:.2%}")
print(f"Avg response time: {metrics_data.avg_response_time:.2f}s")
print(f"Avg tokens: {metrics_data.avg_tokens:.0f}")
print(f"Avg rating: {metrics_data.avg_rating:.1f}/5.0")
print(f"Last used: {metrics_data.last_used}")
```

### Comparing Variants

```python
# Compare all variants in experiment
variant_versions = [v.version for v in experiment.variants]
version_ids = [
    (await version_manager.get_version("greeting", "system", v)).version_id
    for v in variant_versions
]

comparison = await metrics.compare_variants(version_ids)

for version_id, data in comparison.items():
    version = await version_manager.get_version(..., version_id=version_id)
    print(f"\\nVersion {version.version}:")
    print(f"  Uses: {data.total_uses}")
    print(f"  Success: {data.success_rate:.1%}")
    print(f"  Rating: {data.avg_rating:.1f}/5.0")
```

### Finding Top Performers

```python
# Get top versions by success rate
top_by_success = await metrics.get_top_versions(
    version_ids,
    metric="success_rate",
    limit=3
)

for version_id, success_rate in top_by_success:
    print(f"{version_id}: {success_rate:.1%}")

# Get top versions by user rating
top_by_rating = await metrics.get_top_versions(
    version_ids,
    metric="avg_rating",
    limit=3
)

# Get fastest versions (lower response time is better)
fastest = await metrics.get_top_versions(
    version_ids,
    metric="avg_response_time",
    limit=3
)
```

### Summary Statistics

```python
summary = await metrics.get_summary(version_ids)

print(f"Total versions: {summary['total_versions']}")
print(f"Total uses: {summary['total_uses']}")
print(f"Overall success rate: {summary['overall_success_rate']:.1%}")

for version_id, stats in summary['versions'].items():
    print(f"\\n{version_id}:")
    print(f"  Uses: {stats['uses']}")
    print(f"  Success: {stats['success_rate']:.1%}")
```

## Managing Experiments

### Listing Experiments

```python
# All experiments
all_experiments = await ab_manager.list_experiments()

# Filter by name
greeting_experiments = await ab_manager.list_experiments(name="greeting")

# Filter by status
running = await ab_manager.list_experiments(status="running")
completed = await ab_manager.list_experiments(status="completed")
```

### Pausing Experiments

```python
# Pause experiment
await ab_manager.update_experiment_status(
    experiment.experiment_id,
    status="paused"
)

# Resume experiment
await ab_manager.update_experiment_status(
    experiment.experiment_id,
    status="running"
)
```

### Completing Experiments

```python
# Mark as completed (auto-sets end_date)
await ab_manager.update_experiment_status(
    experiment.experiment_id,
    status="completed"
)

# Check experiment duration
experiment = await ab_manager.get_experiment(experiment_id)
duration = experiment.end_date - experiment.start_date
print(f"Experiment ran for {duration.days} days")
```

### Viewing User Assignments

```python
# Get specific user's assignment
assignment = await ab_manager.get_user_assignment(
    experiment.experiment_id,
    user_id="user123"
)

# Get all assignments for experiment
all_assignments = await ab_manager.get_experiment_assignments(
    experiment.experiment_id
)
print(f"Total users: {len(all_assignments)}")

# Get variant distribution
distribution = await ab_manager.get_variant_distribution(
    experiment.experiment_id
)
print("Actual distribution:")
for version, count in distribution.items():
    percentage = (count / sum(distribution.values())) * 100
    print(f"  {version}: {count} users ({percentage:.1f}%)")
```

## Best Practices

### 1. Run Experiments Long Enough

```python
# ❌ Don't: Stop after 1 day
# Not statistically significant

# ✅ Do: Run for appropriate duration
# - Minimum 7 days for user-facing features
# - Ensure statistical significance
# - Consider weekly patterns
```

### 2. Use User-Sticky for Consistency

```python
# ❌ Don't: Use random for user-facing apps
variant = await ab_manager.get_random_variant(experiment_id)
# User sees different variant on each request - confusing!

# ✅ Do: Use sticky assignment
variant = await ab_manager.get_variant_for_user(
    experiment_id,
    user_id=request.user_id
)
# User always sees same variant - consistent experience
```

### 3. Start with 50/50 Split

```python
# ❌ Don't: Start with uneven split without reason
variants=[
    PromptVariant("1.0.0", 0.2, "Control"),
    PromptVariant("1.1.0", 0.8, "Treatment")  # Too risky!
]

# ✅ Do: Start balanced, adjust if needed
variants=[
    PromptVariant("1.0.0", 0.5, "Control"),
    PromptVariant("1.1.0", 0.5, "Treatment")
]
```

### 4. Track Multiple Metrics

```python
# ❌ Don't: Only track success/failure
await metrics.record_event(version_id, success=True)

# ✅ Do: Track comprehensive metrics
await metrics.record_event(
    version_id=version_id,
    success=True,
    response_time=response_time,
    tokens=token_count,
    user_rating=user_feedback,
    metadata={
        "user_segment": "premium",
        "device_type": "mobile"
    }
)
```

### 5. Document Hypotheses

```python
# ❌ Don't: Create experiments without context
experiment = await ab_manager.create_experiment(...)

# ✅ Do: Document purpose and hypothesis
experiment = await ab_manager.create_experiment(
    name="greeting",
    prompt_type="system",
    variants=[...],
    metadata={
        "hypothesis": "Adding user name increases engagement by 20%",
        "owner": "product-team",
        "jira": "EXP-123",
        "success_criteria": "engagement_rate > 0.15",
        "sample_size_target": 10000
    }
)
```

### 6. Clean Up Completed Experiments

```python
# Archive old experiments
import datetime

experiments = await ab_manager.list_experiments(status="completed")
cutoff = datetime.datetime.now() - datetime.timedelta(days=90)

for exp in experiments:
    if exp.end_date and exp.end_date < cutoff:
        await ab_manager.delete_experiment(exp.experiment_id)
```

## Examples

### Example 1: Simple A/B Test

```python
# 1. Create versions
v1 = await version_manager.create_version(
    name="email_subject",
    prompt_type="user",
    template="Generate subject for: {{topic}}",
    version="1.0.0"
)

v2 = await version_manager.create_version(
    name="email_subject",
    prompt_type="user",
    template="Generate engaging subject line for: {{topic}}",
    version="1.1.0"
)

# 2. Create experiment
experiment = await ab_manager.create_experiment(
    name="email_subject",
    prompt_type="user",
    variants=[
        PromptVariant("1.0.0", 0.5, "Basic"),
        PromptVariant("1.1.0", 0.5, "Engaging")
    ]
)

# 3. Run for users
for user_id in users:
    variant = await ab_manager.get_variant_for_user(
        experiment.experiment_id,
        user_id
    )

    # Use variant and track metrics
    version = await version_manager.get_version(
        "email_subject", "user", variant
    )

    # ... use version ...

    await metrics.record_event(
        version.version_id,
        success=email_opened,
        user_rating=user_click_through
    )

# 4. Analyze results
comparison = await metrics.compare_variants([v1.version_id, v2.version_id])

if comparison[v2.version_id].success_rate > comparison[v1.version_id].success_rate:
    print("Treatment wins! Deploying v1.1.0")
    await version_manager.tag_version(v2.version_id, "production")
    await ab_manager.update_experiment_status(experiment.experiment_id, "completed")
```

### Example 2: Multi-Variant Test with Analysis

```python
# Test 3 different tone variations
variants_data = [
    ("1.0.0", "Professional and formal"),
    ("1.1.0", "Friendly and casual"),
    ("1.2.0", "Enthusiastic and energetic")
]

# Create experiment
experiment = await ab_manager.create_experiment(
    name="customer_support",
    prompt_type="system",
    variants=[
        PromptVariant(version, 1/3, desc)
        for version, desc in variants_data
    ]
)

# After running experiment...
version_ids = [
    (await version_manager.get_version("customer_support", "system", v)).version_id
    for v, _ in variants_data
]

# Comprehensive analysis
print("=== Experiment Results ===\\n")

summary = await metrics.get_summary(version_ids)
print(f"Total interactions: {summary['total_uses']}")
print(f"Overall success: {summary['overall_success_rate']:.1%}\\n")

# Rank by rating
top_rated = await metrics.get_top_versions(
    version_ids,
    metric="avg_rating"
)

print("Rankings by user rating:")
for i, (version_id, rating) in enumerate(top_rated, 1):
    version = await version_manager.get_version(
        "customer_support", "system", version_id=version_id
    )
    data = await metrics.get_metrics(version_id)
    print(f"{i}. v{version.version} ({rating:.2f}/5.0)")
    print(f"   Uses: {data.total_uses}, Success: {data.success_rate:.1%}")
```

## See Also

- [Versioning Guide](./VERSIONING.md) - Managing prompt versions
- [User Guide](./USER_GUIDE.md) - General prompt library usage

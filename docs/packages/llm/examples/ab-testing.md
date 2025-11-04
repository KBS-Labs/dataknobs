# A/B Testing Examples

Version management, experiments, and metrics tracking workflows.

## Basic Version Management

### Creating Versions

```python
from dataknobs_llm.prompts import VersionManager, VersionStatus

# Create version manager
vm = VersionManager()

# Create initial version
v1 = await vm.create_version(
    name="email_subject",
    prompt_type="user",
    template="Generate email subject for: {{topic}}",
    version="1.0.0",
    metadata={
        "author": "alice",
        "description": "Initial version",
        "created_for": "email-campaign-q1"
    }
)

print(f"Created version {v1.version} with ID {v1.version_id}")

# Create improved version
v2 = await vm.create_version(
    name="email_subject",
    prompt_type="user",
    template="Generate engaging and click-worthy email subject for: {{topic}}",
    # version="1.0.1" auto-assigned
    metadata={
        "author": "bob",
        "description": "Added 'engaging and click-worthy' for better results",
        "jira_ticket": "PROMPT-123"
    }
)

print(f"Created version {v2.version} (auto-incremented)")
print(f"Parent version: {v2.parent_version}")
```

### Version History

```python
# List all versions
versions = await vm.list_versions("email_subject", "user")

print("Version History:")
for v in versions:
    print(f"  {v.version}: {v.metadata.get('description')}")
    print(f"    Created: {v.created_at}")
    print(f"    Author: {v.metadata.get('author')}")
    print(f"    Status: {v.status.value}")
    print()

# Get specific version
version = await vm.get_version("email_subject", "user", "1.0.0")

# Get latest version
latest = await vm.get_version("email_subject", "user")
print(f"Latest version: {latest.version}")
```

### Version Status Lifecycle

```python
# Create as draft
draft = await vm.create_version(
    name="new_feature",
    prompt_type="system",
    template="...",
    status=VersionStatus.DRAFT,
    metadata={"note": "Still testing"}
)

# Test the draft...
print(f"Testing version {draft.version}...")

# Promote to active
await vm.update_status(draft.version_id, VersionStatus.ACTIVE)
print("Promoted to ACTIVE")

# Tag for production
await vm.tag_version(draft.version_id, "production")
print("Tagged as production")

# Later, deprecate old version
old_version = versions[-2]
await vm.update_status(old_version.version_id, VersionStatus.DEPRECATED)
print(f"Deprecated version {old_version.version}")
```

## A/B Testing

### Simple A/B Test

```python
from dataknobs_llm.prompts import ABTestManager, PromptVariant, MetricsCollector

# Setup
vm = VersionManager()
ab = ABTestManager()
mc = MetricsCollector()

# Create two versions
v1 = await vm.create_version(
    name="greeting",
    prompt_type="system",
    template="Hello {{name}}!",
    version="1.0.0"
)

v2 = await vm.create_version(
    name="greeting",
    prompt_type="system",
    template="Hi {{name}}, welcome to our service!"
)

# Create A/B test (50/50 split)
exp = await ab.create_experiment(
    name="greeting",
    prompt_type="system",
    variants=[
        PromptVariant("1.0.0", 0.5, "Control"),
        PromptVariant("1.0.1", 0.5, "Treatment")
    ],
    metadata={
        "hypothesis": "Friendly greeting increases engagement",
        "start_date": "2024-01-15",
        "owner": "product-team"
    }
)

print(f"Created experiment {exp.experiment_id}")
print(f"Variants: {[(v.version, v.weight) for v in exp.variants]}")
```

### Running the Experiment

```python
from dataknobs_llm import create_llm_provider, LLMConfig
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary
from pathlib import Path

# Setup
config = LLMConfig(provider="openai", api_key="your-key")
llm = create_llm_provider(config)
library = FileSystemPromptLibrary(prompt_dir=Path("prompts/"))
builder = AsyncPromptBuilder(library=library)

# Simulate user sessions
users = ["user1", "user2", "user3", "user4", "user5"]

for user_id in users:
    # Get variant for user (sticky - same user always gets same variant)
    variant_version = await ab.get_variant_for_user(exp.experiment_id, user_id)

    # Get the version
    version = await vm.get_version("greeting", "system", variant_version)

    print(f"\n{user_id} gets version {variant_version}")

    # Use the version
    # (In real app, you'd render the template and use with LLM)
    template_str = version.template.replace("{{name}}", user_id)

    # Track success
    import random
    success = random.random() > 0.3  # Simulate 70% success rate

    await mc.record_event(
        version_id=version.version_id,
        success=success,
        response_time=random.uniform(0.1, 1.0),
        user_rating=random.uniform(3.0, 5.0)
    )

    print(f"  Result: {'Success' if success else 'Failure'}")
```

### Analyzing Results

```python
# Get metrics for each variant
v1_metrics = await mc.get_metrics(v1.version_id)
v2_metrics = await mc.get_metrics(v2.version_id)

print("\nVariant A (Control):")
print(f"  Success rate: {v1_metrics.success_rate:.2%}")
print(f"  Avg response time: {v1_metrics.avg_response_time:.2f}s")
print(f"  Avg rating: {v1_metrics.avg_rating:.1f}/5.0")
print(f"  Total events: {v1_metrics.total_events}")

print("\nVariant B (Treatment):")
print(f"  Success rate: {v2_metrics.success_rate:.2%}")
print(f"  Avg response time: {v2_metrics.avg_response_time:.2f}s")
print(f"  Avg rating: {v2_metrics.avg_rating:.1f}/5.0")
print(f"  Total events: {v2_metrics.total_events}")

# Compare variants
comparison = await mc.compare_variants([v1.version_id, v2.version_id])

print("\nComparison:")
for version_id, metrics in comparison.items():
    version_num = "A" if version_id == v1.version_id else "B"
    print(f"Variant {version_num}: {metrics.success_rate:.2%} success rate")
```

### Deploy Winner

```python
# Determine winner
if v2_metrics.success_rate > v1_metrics.success_rate:
    winner = v2
    print(f"\nWinner: Variant B (v{v2.version})")
    print(f"Improvement: {(v2_metrics.success_rate - v1_metrics.success_rate):.2%}")

    # Deploy winner
    await vm.tag_version(v2.version_id, "production")
    await vm.update_status(v2.version_id, VersionStatus.PRODUCTION)

    # Mark experiment complete
    await ab.update_experiment_status(exp.experiment_id, "completed")

    # Deprecate old version
    await vm.update_status(v1.version_id, VersionStatus.DEPRECATED)

    print("Winner deployed to production!")
else:
    print("\nControl wins - keeping current version")
    await vm.tag_version(v1.version_id, "production")
```

## Multi-Variant Testing (A/B/C)

### Three-Way Split

```python
# Create three versions
v1 = await vm.create_version(
    name="ad_copy",
    prompt_type="user",
    template="Buy {{product}} now!",
    version="1.0.0"
)

v2 = await vm.create_version(
    name="ad_copy",
    prompt_type="user",
    template="Get {{product}} today with free shipping!"
)

v3 = await vm.create_version(
    name="ad_copy",
    prompt_type="user",
    template="Limited offer: {{product}} at 20% off!"
)

# Create A/B/C test (equal split)
exp = await ab.create_experiment(
    name="ad_copy",
    prompt_type="user",
    variants=[
        PromptVariant("1.0.0", 0.33, "Direct"),
        PromptVariant("1.0.1", 0.33, "Value"),
        PromptVariant("1.0.2", 0.34, "Urgency")
    ]
)

# Run experiment and analyze
# ... (similar to A/B test)

# Find top performer
top_versions = await mc.get_top_versions(
    version_ids=[v1.version_id, v2.version_id, v3.version_id],
    metric="success_rate",
    limit=1
)

winner_id, winner_metrics = top_versions[0]
print(f"Winner: {winner_id} with {winner_metrics.success_rate:.2%} success rate")
```

## Advanced Patterns

### Gradual Rollout

```python
# Phase 1: Test with 10% of traffic
exp_phase1 = await ab.create_experiment(
    name="new_feature",
    prompt_type="system",
    variants=[
        PromptVariant("1.0.0", 0.9, "Current"),
        PromptVariant("2.0.0", 0.1, "New")  # Only 10% get new version
    ],
    metadata={"phase": "1", "rollout_strategy": "gradual"}
)

# Monitor for 7 days...
await asyncio.sleep(7 * 24 * 3600)  # In reality, this would be monitoring

# Check metrics
new_metrics = await mc.get_metrics(v_new.version_id)

if new_metrics.success_rate >= 0.95:  # Good performance
    # Phase 2: Increase to 50%
    exp_phase2 = await ab.create_experiment(
        name="new_feature",
        prompt_type="system",
        variants=[
            PromptVariant("1.0.0", 0.5, "Current"),
            PromptVariant("2.0.0", 0.5, "New")
        ],
        metadata={"phase": "2"}
    )

    # Monitor again...

    # Phase 3: Full rollout
    await vm.tag_version(v_new.version_id, "production")
else:
    # Rollback
    print("New version underperforming, stopping rollout")
    await ab.update_experiment_status(exp_phase1.experiment_id, "stopped")
```

### Segment-Based Testing

```python
# Test different variants for different user segments
class SegmentedABTest:
    def __init__(self, ab_manager, vm):
        self.ab = ab_manager
        self.vm = vm

    async def get_variant_for_segment(self, experiment_id, user_id, segment):
        # Different experiments for different segments
        segment_exp_id = f"{experiment_id}_{segment}"

        variant = await self.ab.get_variant_for_user(segment_exp_id, user_id)
        return variant

# Create segment-specific experiments
premium_exp = await ab.create_experiment(
    name="greeting_premium",
    prompt_type="system",
    variants=[
        PromptVariant("1.0.0", 0.5, "Formal"),
        PromptVariant("1.1.0", 0.5, "VIP")
    ]
)

free_exp = await ab.create_experiment(
    name="greeting_free",
    prompt_type="system",
    variants=[
        PromptVariant("1.0.0", 0.5, "Standard"),
        PromptVariant("1.2.0", 0.5, "Encouraging")
    ]
)

# Use based on user segment
def get_segment(user_id):
    # Determine user segment
    return "premium" if user_id.startswith("premium") else "free"

segment = get_segment(user_id)
exp_id = f"greeting_{segment}"
variant = await ab.get_variant_for_user(exp_id, user_id)
```

### Real-Time Metrics Tracking

```python
import asyncio
from datetime import datetime, timedelta

class RealTimeMetricsMonitor:
    def __init__(self, metrics_collector):
        self.mc = metrics_collector
        self.alerts = []

    async def monitor_experiment(
        self,
        version_ids,
        duration_hours=24,
        check_interval_minutes=30,
        alert_threshold=0.5  # Alert if success rate < 50%
    ):
        end_time = datetime.utcnow() + timedelta(hours=duration_hours)

        while datetime.utcnow() < end_time:
            print(f"\n=== Metrics Check at {datetime.utcnow()} ===")

            for version_id in version_ids:
                metrics = await self.mc.get_metrics(version_id)

                print(f"\nVersion {version_id}:")
                print(f"  Events: {metrics.total_events}")
                print(f"  Success Rate: {metrics.success_rate:.2%}")
                print(f"  Avg Response Time: {metrics.avg_response_time:.2f}s")

                # Check for alerts
                if metrics.success_rate < alert_threshold:
                    alert = f"⚠️  Version {version_id} below threshold: {metrics.success_rate:.2%}"
                    print(alert)
                    self.alerts.append({
                        "time": datetime.utcnow(),
                        "version_id": version_id,
                        "message": alert
                    })

            # Wait before next check
            await asyncio.sleep(check_interval_minutes * 60)

        print("\n=== Monitoring Complete ===")
        return self.alerts

# Use the monitor
monitor = RealTimeMetricsMonitor(mc)
alerts = await monitor.monitor_experiment(
    version_ids=[v1.version_id, v2.version_id],
    duration_hours=24,
    check_interval_minutes=30
)

if alerts:
    print(f"\n{len(alerts)} alerts triggered during experiment")
```

### Statistical Significance

```python
from scipy import stats

async def check_statistical_significance(mc, v1_id, v2_id, alpha=0.05):
    """Check if difference between variants is statistically significant."""

    # Get metrics
    m1 = await mc.get_metrics(v1_id)
    m2 = await mc.get_metrics(v2_id)

    # Need raw event data for proper test
    # This is simplified - in production, use proper A/B test statistics
    n1 = m1.total_events
    n2 = m2.total_events
    p1 = m1.success_rate
    p2 = m2.success_rate

    # Z-test for proportions
    p_pooled = (n1 * p1 + n2 * p2) / (n1 + n2)
    se = (p_pooled * (1 - p_pooled) * (1/n1 + 1/n2)) ** 0.5
    z = (p2 - p1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    print(f"\nStatistical Analysis:")
    print(f"  Variant A: {p1:.2%} ({n1} events)")
    print(f"  Variant B: {p2:.2%} ({n2} events)")
    print(f"  Difference: {abs(p2 - p1):.2%}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significant at α={alpha}: {p_value < alpha}")

    return {
        "significant": p_value < alpha,
        "p_value": p_value,
        "winner": "B" if p2 > p1 else "A"
    }

# Check significance before deploying
result = await check_statistical_significance(mc, v1.version_id, v2.version_id)

if result["significant"]:
    print(f"\n✓ Variant {result['winner']} is significantly better")
    # Deploy winner
else:
    print("\n✗ No significant difference - need more data")
    # Continue experiment
```

## Rollback Workflow

### Safe Rollback

```python
async def safe_rollback(vm, mc, current_version_id, experiment_id):
    """Rollback to previous version if current version is failing."""

    # Check current version metrics
    current_metrics = await mc.get_metrics(current_version_id)

    print(f"Current version metrics:")
    print(f"  Success rate: {current_metrics.success_rate:.2%}")
    print(f"  Events: {current_metrics.total_events}")

    # Define rollback criteria
    if current_metrics.success_rate < 0.8 and current_metrics.total_events > 100:
        print("\n⚠️  Current version underperforming - initiating rollback")

        # Get current version
        current_version = await vm.get_version_by_id(current_version_id)

        # Get parent version (previous)
        if current_version.parent_version:
            parent_version = await vm.get_version_by_id(current_version.parent_version)

            print(f"Rolling back to version {parent_version.version}")

            # Update tags and status
            await vm.untag_version(current_version_id, "production")
            await vm.tag_version(parent_version.version_id, "production")
            await vm.update_status(current_version_id, VersionStatus.DEPRECATED)
            await vm.update_status(parent_version.version_id, VersionStatus.PRODUCTION)

            # Stop experiment
            await ab.update_experiment_status(experiment_id, "stopped")

            print("✓ Rollback complete")
            return parent_version
        else:
            print("✗ No parent version to rollback to")
            return None
    else:
        print("✓ Current version performing well")
        return None

# Monitor and rollback if needed
rolled_back = await safe_rollback(
    vm, mc, v_new.version_id, exp.experiment_id
)
```

## See Also

- [Versioning & A/B Testing Guide](../guides/versioning.md) - Detailed guide
- [Versioning API](../api/versioning.md) - Complete API reference
- [Performance Guide](../guides/performance.md) - Optimization tips
- [Basic Usage Examples](basic-usage.md) - Getting started

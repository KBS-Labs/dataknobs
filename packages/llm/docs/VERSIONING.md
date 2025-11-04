# Prompt Versioning Guide

This guide covers the prompt versioning system in dataknobs-llm, which provides version control, A/B testing, and metrics tracking for prompts.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Version Management](#version-management)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)
- [Examples](#examples)

## Overview

The versioning system provides:

- **Semantic Versioning**: Track prompt versions with `major.minor.patch` format
- **Version History**: See the evolution of prompts over time
- **Status Management**: Mark versions as draft, active, production, deprecated, or archived
- **Tag Support**: Tag versions for easy filtering (e.g., "production", "experimental")
- **Rollback**: Easy rollback to previous versions
- **Parent Tracking**: Maintain version lineage

## Quick Start

```python
from dataknobs_llm.prompts import VersionManager

# Create version manager
manager = VersionManager()

# Create first version
v1 = await manager.create_version(
    name="greeting",
    prompt_type="system",
    template="Hello {{name}}!",
    version="1.0.0",
    metadata={"author": "alice", "purpose": "Initial greeting"}
)

# Create new version (auto-increments to 1.0.1)
v2 = await manager.create_version(
    name="greeting",
    prompt_type="system",
    template="Hello {{name}}, welcome to {{app_name}}!",
    # version not specified - auto-increments
    metadata={"author": "bob", "purpose": "Add app name"}
)

# Get latest version
latest = await manager.get_version("greeting", "system", version="latest")
print(f"Latest version: {latest.version}")  # "1.0.1"

# Get specific version
v1_retrieved = await manager.get_version("greeting", "system", version="1.0.0")

# Tag a version as production
await manager.tag_version(v2.version_id, "production")

# List all versions
versions = await manager.list_versions("greeting", "system")
for v in versions:
    print(f"v{v.version}: {v.template[:50]}...")

# Get version history (chronological)
history = await manager.get_version_history("greeting", "system")
```

## Version Management

### Creating Versions

#### With Explicit Version Number

```python
version = await manager.create_version(
    name="code_analysis",
    prompt_type="system",
    template="Analyze this {{language}} code: {{code}}",
    version="1.0.0",
    defaults={"language": "python"},
    metadata={
        "author": "alice",
        "description": "Initial code analysis prompt",
        "created_date": "2025-01-15"
    }
)
```

#### With Auto-Increment

```python
# Automatically increments from latest version
v2 = await manager.create_version(
    name="code_analysis",
    prompt_type="system",
    template="Analyze this {{language}} code and provide suggestions: {{code}}",
    # version="1.0.1" automatically assigned
)
```

### Retrieving Versions

#### Get Latest Version

```python
latest = await manager.get_version(
    name="code_analysis",
    prompt_type="system",
    version="latest"  # or omit version parameter
)
```

#### Get Specific Version

```python
v1 = await manager.get_version(
    name="code_analysis",
    prompt_type="system",
    version="1.0.0"
)
```

#### Get by Version ID

```python
version = await manager.get_version(
    name="code_analysis",
    prompt_type="system",
    version_id="abc-123-def-456"
)
```

### Listing Versions

#### All Versions

```python
versions = await manager.list_versions("code_analysis", "system")
# Returns versions sorted newest first
```

#### Filter by Tags

```python
prod_versions = await manager.list_versions(
    name="code_analysis",
    prompt_type="system",
    tags=["production"]
)
```

#### Filter by Status

```python
from dataknobs_llm.prompts import VersionStatus

active_versions = await manager.list_versions(
    name="code_analysis",
    prompt_type="system",
    status=VersionStatus.PRODUCTION
)
```

## Core Concepts

### Semantic Versioning

Versions follow semantic versioning (`major.minor.patch`):

- **Major** (X.0.0): Breaking changes or complete rewrites
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, minor improvements

```python
# Major version change
v2 = await manager.create_version(
    name="greeting",
    prompt_type="system",
    template="Completely new greeting system",
    version="2.0.0"
)

# Minor version change
v1_1 = await manager.create_version(
    name="greeting",
    prompt_type="system",
    template="Hello {{name}}, {{extra_info}}",
    version="1.1.0"
)

# Patch version change
v1_0_1 = await manager.create_version(
    name="greeting",
    prompt_type="system",
    template="Hello {{name}}.",  # Fixed punctuation
    version="1.0.1"
)
```

### Version Status

Each version has a status indicating its lifecycle stage:

```python
from dataknobs_llm.prompts import VersionStatus

# Create as draft
draft = await manager.create_version(
    name="new_feature",
    prompt_type="user",
    template="Work in progress...",
    version="1.0.0",
    status=VersionStatus.DRAFT
)

# Promote to active
await manager.update_status(draft.version_id, VersionStatus.ACTIVE)

# Promote to production
await manager.update_status(draft.version_id, VersionStatus.PRODUCTION)

# Deprecate old version
await manager.update_status(old_version_id, VersionStatus.DEPRECATED)

# Archive obsolete version
await manager.update_status(old_version_id, VersionStatus.ARCHIVED)
```

**Status Lifecycle**:
```
DRAFT → ACTIVE → PRODUCTION → DEPRECATED → ARCHIVED
```

### Tags

Tags provide flexible categorization:

```python
# Tag a version
await manager.tag_version(version_id, "production")
await manager.tag_version(version_id, "tested")
await manager.tag_version(version_id, "approved-by-legal")

# Remove a tag
await manager.untag_version(version_id, "production")

# List versions with specific tags
prod_versions = await manager.list_versions(
    name="greeting",
    prompt_type="system",
    tags=["production", "approved"]  # Versions with ANY of these tags
)
```

**Common Tags**:
- `production` - Currently in production
- `staging` - Testing in staging environment
- `experimental` - Experimental features
- `approved` - Reviewed and approved
- `deprecated` - Scheduled for removal

### Parent Tracking

Versions maintain lineage through parent references:

```python
# Create base version
v1 = await manager.create_version(
    name="prompt",
    prompt_type="system",
    template="V1",
    version="1.0.0"
)

# Create child version (parent auto-set)
v2 = await manager.create_version(
    name="prompt",
    prompt_type="system",
    template="V2"
)

print(v2.parent_version)  # v1.version_id

# Explicit parent setting
v3 = await manager.create_version(
    name="prompt",
    prompt_type="system",
    template="V3",
    parent_version=v1.version_id  # Skip v2, derive from v1
)
```

## API Reference

### VersionManager

#### Constructor

```python
manager = VersionManager(storage=None)
```

- `storage`: Optional backend storage (None for in-memory)

#### create_version()

```python
version = await manager.create_version(
    name: str,
    prompt_type: str,
    template: str,
    version: Optional[str] = None,
    defaults: Optional[Dict[str, Any]] = None,
    validation: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    created_by: Optional[str] = None,
    parent_version: Optional[str] = None,
    tags: Optional[List[str]] = None,
    status: VersionStatus = VersionStatus.ACTIVE,
) -> PromptVersion
```

#### get_version()

```python
version = await manager.get_version(
    name: str,
    prompt_type: str,
    version: str = "latest",
    version_id: Optional[str] = None,
) -> Optional[PromptVersion]
```

#### list_versions()

```python
versions = await manager.list_versions(
    name: str,
    prompt_type: str,
    tags: Optional[List[str]] = None,
    status: Optional[VersionStatus] = None,
) -> List[PromptVersion]
```

#### tag_version() / untag_version()

```python
version = await manager.tag_version(version_id: str, tag: str) -> PromptVersion
version = await manager.untag_version(version_id: str, tag: str) -> PromptVersion
```

#### update_status()

```python
version = await manager.update_status(
    version_id: str,
    status: VersionStatus,
) -> PromptVersion
```

#### delete_version()

```python
deleted = await manager.delete_version(version_id: str) -> bool
```

#### get_version_history()

```python
history = await manager.get_version_history(
    name: str,
    prompt_type: str,
) -> List[PromptVersion]
```

### PromptVersion

```python
@dataclass
class PromptVersion:
    version_id: str
    name: str
    prompt_type: str
    version: str
    template: str
    defaults: Dict[str, Any]
    validation: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: datetime
    created_by: Optional[str]
    parent_version: Optional[str]
    tags: List[str]
    status: VersionStatus
```

## Best Practices

### 1. Use Semantic Versioning Correctly

```python
# ❌ Don't: Use arbitrary version numbers
version="5"  # What does this mean?

# ✅ Do: Follow semantic versioning
version="1.0.0"  # Clear: first release
version="1.1.0"  # Clear: added feature
version="2.0.0"  # Clear: breaking change
```

### 2. Always Add Metadata

```python
# ❌ Don't: Create versions without context
await manager.create_version(
    name="prompt",
    prompt_type="system",
    template="..."
)

# ✅ Do: Include descriptive metadata
await manager.create_version(
    name="prompt",
    prompt_type="system",
    template="...",
    metadata={
        "author": "alice",
        "description": "Fixed hallucination issue",
        "jira_ticket": "PROMPT-123",
        "created_date": "2025-01-15",
        "tested": True
    },
    created_by="alice@example.com"
)
```

### 3. Use Status Lifecycle

```python
# ❌ Don't: Deploy directly to production
v = await manager.create_version(..., status=VersionStatus.PRODUCTION)

# ✅ Do: Follow the lifecycle
# 1. Create as draft
v = await manager.create_version(..., status=VersionStatus.DRAFT)

# 2. Test and activate
await manager.update_status(v.version_id, VersionStatus.ACTIVE)

# 3. Deploy to production
await manager.update_status(v.version_id, VersionStatus.PRODUCTION)

# 4. Eventually deprecate
await manager.update_status(v.version_id, VersionStatus.DEPRECATED)
```

### 4. Tag Production Versions

```python
# Mark current production version
await manager.tag_version(version_id, "production")

# When deploying new version
await manager.untag_version(old_version_id, "production")
await manager.tag_version(new_version_id, "production")
```

### 5. Use Auto-Increment for Minor Changes

```python
# ❌ Don't: Manually track patch versions
v1 = await manager.create_version(..., version="1.0.0")
v2 = await manager.create_version(..., version="1.0.1")  # Easy to forget
v3 = await manager.create_version(..., version="1.0.2")  # Prone to errors

# ✅ Do: Let the system auto-increment
v1 = await manager.create_version(..., version="1.0.0")
v2 = await manager.create_version(...)  # Auto: 1.0.1
v3 = await manager.create_version(...)  # Auto: 1.0.2
```

### 6. Archive Old Versions

```python
# Get versions older than 6 months
old_versions = await manager.list_versions("prompt", "system")
cutoff_date = datetime.now() - timedelta(days=180)

for version in old_versions:
    if version.created_at < cutoff_date and version.status != VersionStatus.PRODUCTION:
        await manager.update_status(version.version_id, VersionStatus.ARCHIVED)
```

## Examples

### Example 1: Version Evolution

```python
# Month 1: Initial version
v1_0_0 = await manager.create_version(
    name="customer_support",
    prompt_type="system",
    template="You are a helpful customer support agent. Answer: {{question}}",
    version="1.0.0",
    metadata={"milestone": "launch"}
)

# Month 2: Add tone guidance
v1_1_0 = await manager.create_version(
    name="customer_support",
    prompt_type="system",
    template="You are a friendly customer support agent. Be empathetic. Answer: {{question}}",
    version="1.1.0",
    metadata={"improvement": "added tone guidance"}
)

# Month 2: Fix typo in 1.1.0
v1_1_1 = await manager.create_version(
    name="customer_support",
    prompt_type="system",
    template="You are a friendly customer support agent. Be empathetic. Answer: {{question}}",
    version="1.1.1",
    metadata={"fix": "typo correction"}
)

# Month 6: Major rewrite
v2_0_0 = await manager.create_version(
    name="customer_support",
    prompt_type="system",
    template="{{system_role}}\\n\\nUser query: {{question}}\\n\\nGuidelines: {{guidelines}}",
    version="2.0.0",
    metadata={"milestone": "major rewrite with templates"}
)
```

### Example 2: Rollback Scenario

```python
# Deploy new version
new_version = await manager.create_version(
    name="greeting",
    prompt_type="system",
    template="New greeting with bug",
    version="2.0.0"
)
await manager.tag_version(new_version.version_id, "production")

# Monitor and detect issues...
# Need to rollback!

# Get previous production version
versions = await manager.list_versions("greeting", "system")
previous = versions[1]  # Second newest (first is current)

# Rollback
await manager.untag_version(new_version.version_id, "production")
await manager.tag_version(previous.version_id, "production")
await manager.update_status(new_version.version_id, VersionStatus.DEPRECATED)

# Use the rollback version
library.get_system_prompt("greeting", version=previous.version)
```

### Example 3: Team Collaboration

```python
# Alice creates initial version
v1 = await manager.create_version(
    name="code_review",
    prompt_type="system",
    template="Review this code: {{code}}",
    version="1.0.0",
    created_by="alice@example.com",
    metadata={"reviewer": "alice"}
)

# Bob improves it
v2 = await manager.create_version(
    name="code_review",
    prompt_type="system",
    template="Review this {{language}} code for bugs and style: {{code}}",
    created_by="bob@example.com",
    metadata={
        "reviewer": "bob",
        "changes": "added language context and specific review criteria"
    }
)

# View history
history = await manager.get_version_history("code_review", "system")
for v in history:
    print(f"v{v.version} by {v.created_by}: {v.metadata.get('changes', 'initial')}")
```

## See Also

- [A/B Testing Guide](./AB_TESTING.md) - Running experiments with versions
- [Metrics Guide](./AB_TESTING.md#metrics-tracking) - Tracking version performance
- [User Guide](./USER_GUIDE.md) - General prompt library usage

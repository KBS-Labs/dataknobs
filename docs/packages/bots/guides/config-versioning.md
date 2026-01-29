# Configuration Versioning

Configuration versioning provides version tracking and management for bot configurations, enabling version history, rollback, and audit trails.

## Overview

The versioning system supports:

- **Immutable versions** - Each version is a snapshot, never modified
- **Replace-only updates** - New versions replace, not modify
- **Version history** - Complete audit trail with timestamps
- **Rollback** - Restore previous configurations
- **Diff** - Compare versions to see changes

## Quick Start

```python
from dataknobs_bots.config import ConfigVersionManager, ConfigVersion

# Create manager
manager = ConfigVersionManager()

# Create initial version
v1 = manager.create(
    config={"name": "MyBot", "llm": {"model": "gpt-4"}},
    reason="Initial configuration",
)

# Update creates new version
v2 = manager.update(
    config={"name": "MyBot", "llm": {"model": "gpt-4o"}},
    reason="Upgrade to GPT-4o",
)

# Rollback to previous version
v3 = manager.rollback(to_version=1, reason="Reverting upgrade")

# Get history
for version in manager.get_history():
    print(f"v{version.version}: {version.reason}")
```

## ConfigVersion

Each version is an immutable snapshot:

```python
from dataknobs_bots.config import ConfigVersion

# Versions are created by the manager, but you can inspect them:
version = manager.get_version(1)

print(version.version)           # 1
print(version.config)            # {"name": "MyBot", ...}
print(version.timestamp)         # Unix timestamp
print(version.reason)            # "Initial configuration"
print(version.previous_version)  # None (for v1)
print(version.created_by)        # "configbot"
print(version.metadata)          # {"source": "wizard"}
```

### Fields

| Field | Description |
|-------|-------------|
| `version` | Version number (1-indexed) |
| `config` | Configuration data at this version |
| `timestamp` | When created (Unix timestamp) |
| `reason` | Why this version was created |
| `previous_version` | Version number this derived from |
| `created_by` | Who/what created this version |
| `metadata` | Additional metadata |

## ConfigVersionManager

The main class for version management:

### Creating the Initial Version

```python
manager = ConfigVersionManager()

v1 = manager.create(
    config={
        "name": "SupportBot",
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
        },
        "memory": {
            "type": "buffer",
            "max_messages": 10,
        },
    },
    reason="Initial bot configuration",
    created_by="configbot",
    metadata={"source": "wizard"},
)

# Can only call create() once
# manager.create(...) would raise ValueError
```

### Updating Configuration

```python
# Update creates a new version
v2 = manager.update(
    config={
        "name": "SupportBot",
        "llm": {
            "provider": "anthropic",  # Changed
            "model": "claude-3-sonnet",  # Changed
        },
        "memory": {
            "type": "buffer",
            "max_messages": 10,
        },
    },
    reason="Switch to Claude",
    created_by="admin",
)

assert v2.version == 2
assert v2.previous_version == 1
```

### Optimistic Locking

Prevent concurrent modification conflicts:

```python
from dataknobs_bots.config import VersionConflictError

# Specify expected version
try:
    v3 = manager.update(
        config=new_config,
        reason="My update",
        expected_version=2,  # Expect current to be v2
    )
except VersionConflictError as e:
    print(f"Conflict: expected v{e.expected_version}, got v{e.actual_version}")
    # Another update happened, reload and retry
```

### Rollback

Rollback creates a new version with old config:

```python
# Current is v3, rollback to v1
v4 = manager.rollback(
    to_version=1,
    reason="Reverting to stable configuration",
    created_by="admin",
)

assert v4.version == 4
assert v4.config == manager.get_version(1).config
assert v4.metadata["rollback_from"] == 3
assert v4.metadata["rollback_to"] == 1
```

### Querying Versions

```python
# Get specific version
v2 = manager.get_version(2)

# Get current version number
current_version = manager.current_version

# Get current config
current_config = manager.current_config

# Get history (newest first)
history = manager.get_history()

# Limit results
recent = manager.get_history(limit=5)

# Versions since a specific version
new_versions = manager.get_history(since_version=2)

# Number of versions
count = len(manager)
```

### Comparing Versions

```python
# Get diff between versions
diff = manager.diff(from_version=1, to_version=2)

print(diff["added"])    # Keys added in v2
print(diff["removed"])  # Keys removed in v2
print(diff["changed"])  # Keys with changed values

# Example output:
# {
#     "from_version": 1,
#     "to_version": 2,
#     "added": {},
#     "removed": {},
#     "changed": {
#         "llm": {
#             "from": {"provider": "openai", "model": "gpt-4"},
#             "to": {"provider": "anthropic", "model": "claude-3-sonnet"}
#         }
#     }
# }
```

## Serialization

Save and restore version manager state:

```python
# Serialize to dict
state = manager.to_dict()

# Save to file, database, etc.
import json
with open("versions.json", "w") as f:
    json.dump(state, f)

# Restore from dict
restored_manager = ConfigVersionManager.from_dict(state)
```

## Integration with ConfigBot

ConfigBot uses versioning when saving configurations:

```python
# In ConfigBot save workflow
class SaveConfigTool:
    def __init__(self, version_manager: ConfigVersionManager):
        self.version_manager = version_manager

    async def execute(self, config: dict, reason: str = "User save"):
        if self.version_manager.current_version == 0:
            version = self.version_manager.create(
                config=config,
                reason=reason,
                created_by="configbot",
            )
        else:
            version = self.version_manager.update(
                config=config,
                reason=reason,
                created_by="configbot",
            )

        return {
            "version": version.version,
            "saved_at": version.timestamp,
        }
```

## Use Cases

### Audit Trail

```python
# Track all configuration changes
for version in manager.get_history():
    print(f"v{version.version} at {version.timestamp}")
    print(f"  By: {version.created_by}")
    print(f"  Reason: {version.reason}")
    if version.previous_version:
        diff = manager.diff(version.previous_version, version.version)
        if diff["changed"]:
            print(f"  Changed: {list(diff['changed'].keys())}")
```

### Safe Experimentation

```python
# Save current state
stable_version = manager.current_version

# Try experimental config
try:
    manager.update(experimental_config, reason="Experiment")
    # Test the new config
    result = await test_config(manager.current_config)
    if not result.success:
        raise ValueError("Experiment failed")
except Exception:
    # Rollback to stable
    manager.rollback(to_version=stable_version, reason="Experiment failed")
```

### Version Comparison UI

```python
# For a diff view in UI
def get_version_comparison(manager, v1, v2):
    diff = manager.diff(v1, v2)

    return {
        "from": {
            "version": v1,
            "config": manager.get_version(v1).config,
        },
        "to": {
            "version": v2,
            "config": manager.get_version(v2).config,
        },
        "changes": diff,
    }
```

## Best Practices

1. **Always provide reasons** - Document why each version was created
2. **Use created_by** - Track who/what made changes
3. **Test before committing** - Validate configs before creating versions
4. **Use optimistic locking** - Prevent concurrent modification issues
5. **Rollback carefully** - Test after rollback to ensure stability
6. **Persist regularly** - Save manager state to durable storage

## Error Handling

```python
from dataknobs_bots.config import VersionConflictError

# Handle version conflicts
try:
    manager.update(config, expected_version=current)
except VersionConflictError as e:
    # Reload current state and retry or notify user
    pass

# Handle missing versions
version = manager.get_version(999)
if version is None:
    print("Version not found")

# Handle rollback to non-existent version
try:
    manager.rollback(to_version=999)
except ValueError as e:
    print(f"Rollback failed: {e}")
```

## Limitations

- In-memory storage (implement persistence for production)
- Top-level diff only (nested changes shown as whole object)
- No merge support (replace-only semantics)
- No branching (linear version history)

## Related Documentation

- [Artifact System](artifacts.md) - Artifact versioning with lineage
- [Configuration Reference](configuration.md) - Bot configuration options

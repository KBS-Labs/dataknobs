# Conversation Storage Schema Versioning

**Package**: `dataknobs_llm.conversations`
**Current Schema Version**: 1.0.0
**Last Updated**: 2025-10-29

---

## Overview

The conversation storage system uses **semantic versioning** (MAJOR.MINOR.PATCH) to track and manage schema changes over time. This ensures backward compatibility and provides a clear migration path when the storage format evolves.

## Version Format

Schema versions follow semantic versioning:

- **MAJOR**: Incompatible changes requiring migration
- **MINOR**: Backward-compatible additions (new optional fields)
- **PATCH**: Bug fixes with no schema changes

**Current version**: `1.0.0`

## How It Works

### Automatic Versioning

Every `ConversationState` includes a `schema_version` field:

```python
from dataknobs_llm.conversations import ConversationState, SCHEMA_VERSION

state = ConversationState(
    conversation_id="conv-123",
    message_tree=tree
)

print(state.schema_version)  # "1.0.0"
print(SCHEMA_VERSION)  # "1.0.0" (current version constant)
```

### Automatic Migration on Load

When loading conversations, the system automatically migrates from older schema versions:

```python
# Load old conversation (e.g., schema 0.0.0 or missing version)
state = await storage.load_conversation(conversation_id)

# Automatically migrated to current version
print(state.schema_version)  # "1.0.0"
```

### Migration Logging

Migrations are logged for monitoring:

```python
import logging

logging.basicConfig(level=logging.INFO)

# Load old conversation
state = ConversationState.from_dict(old_data)

# Logs: "Migrating conversation conv-123 from schema 0.0.0 to 1.0.0"
```

## Version History

### Version 1.0.0 (2025-10-29)

**Initial versioned schema**

- Added `schema_version` field to `ConversationState`
- Tree-based conversation structure with nodes and edges
- Support for conversation branching
- Metadata at conversation and node levels

**Migration from unversioned (0.0.0)**:
- No data structure changes
- Simply adds `schema_version: "1.0.0"` field
- All existing conversations load without modification

### Version 0.0.0 (Legacy)

**Unversioned schema** - Initial implementation before versioning was added.

- All conversations created before schema versioning
- Missing `schema_version` field
- Automatically upgraded to 1.0.0 on load

## Adding New Schema Versions

When the storage format needs to change, follow this process:

### 1. Increment Version Number

Update `SCHEMA_VERSION` in `storage.py`:

```python
# Before
SCHEMA_VERSION = "1.0.0"

# After (example: adding optional field)
SCHEMA_VERSION = "1.1.0"
```

### 2. Add Migration Method

Add a static migration method to `ConversationState`:

```python
@staticmethod
def _migrate_1_0_to_1_1(data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate from schema 1.0 to 1.1.

    Changes:
    - Added optional 'tags' field to conversation metadata
    """
    # Add new field with default value
    if "tags" not in data.get("metadata", {}):
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["tags"] = []

    data["schema_version"] = "1.1.0"
    return data
```

### 3. Register Migration in _migrate_schema

Update the `_migrate_schema` method:

```python
@staticmethod
def _migrate_schema(
    data: Dict[str, Any],
    from_version: str,
    to_version: str
) -> Dict[str, Any]:
    """Migrate data from one schema version to another."""
    # ... existing code ...

    # Apply migrations sequentially
    if from_version < "1.1.0" and to_version >= "1.1.0":
        data = ConversationState._migrate_1_0_to_1_1(data)
        from_version = "1.1.0"

    if from_version < "1.2.0" and to_version >= "1.2.0":
        data = ConversationState._migrate_1_1_to_1_2(data)
        from_version = "1.2.0"

    # ... rest of method ...
```

### 4. Add Tests

Create tests for the new migration:

```python
def test_migration_from_1_0_to_1_1():
    """Test migration from 1.0.0 to 1.1.0."""
    data = {
        "schema_version": "1.0.0",
        "conversation_id": "test-123",
        # ... old format data ...
    }

    state = ConversationState.from_dict(data)

    assert state.schema_version == "1.1.0"
    assert "tags" in state.metadata  # New field added
```

### 5. Update Documentation

Document the change:
- Add version entry to this file
- Update CHANGELOG
- Note any breaking changes
- Provide migration examples if needed

## Migration Examples

### Example 1: Adding Optional Field (Minor Version)

**Version 1.0.0 → 1.1.0**: Add optional `tags` field

```python
@staticmethod
def _migrate_1_0_to_1_1(data: Dict[str, Any]) -> Dict[str, Any]:
    """Add tags field to metadata."""
    if "metadata" not in data:
        data["metadata"] = {}

    if "tags" not in data["metadata"]:
        data["metadata"]["tags"] = []

    data["schema_version"] = "1.1.0"
    return data
```

**Backward compatible**: Old conversations gain empty tags array.

### Example 2: Restructuring Data (Major Version)

**Version 1.x → 2.0.0**: Change node ID format

```python
@staticmethod
def _migrate_1_x_to_2_0(data: Dict[str, Any]) -> Dict[str, Any]:
    """Change node ID format from dot-delimited to UUID."""
    import uuid

    # Map old IDs to new UUIDs
    id_mapping = {"": str(uuid.uuid4())}  # Root

    # Update node IDs
    for node in data["nodes"]:
        old_id = node["node_id"]
        if old_id not in id_mapping:
            id_mapping[old_id] = str(uuid.uuid4())
        node["node_id"] = id_mapping[old_id]

    # Update edges
    new_edges = []
    for parent_id, child_id in data["edges"]:
        new_edges.append([
            id_mapping.get(parent_id, parent_id),
            id_mapping.get(child_id, child_id)
        ])
    data["edges"] = new_edges

    # Update current node ID
    data["current_node_id"] = id_mapping.get(
        data["current_node_id"],
        data["current_node_id"]
    )

    data["schema_version"] = "2.0.0"
    return data
```

**Breaking change**: Major version bump required.

### Example 3: Renaming Field (Major Version)

**Version 2.0 → 3.0**: Rename `metadata` to `attributes`

```python
@staticmethod
def _migrate_2_x_to_3_0(data: Dict[str, Any]) -> Dict[str, Any]:
    """Rename metadata field to attributes."""
    if "metadata" in data:
        data["attributes"] = data.pop("metadata")
    else:
        data["attributes"] = {}

    # Also update node metadata
    for node in data.get("nodes", []):
        if "metadata" in node:
            node["attributes"] = node.pop("metadata")
        else:
            node["attributes"] = {}

    data["schema_version"] = "3.0.0"
    return data
```

**Breaking change**: Field renamed, major version bump.

## Error Handling

### Downgrade Protection

Attempting to load a conversation from a future version raises an error:

```python
from dataknobs_llm.conversations import SchemaVersionError

try:
    # Try to load conversation saved with version 2.0.0
    # when current version is 1.0.0
    state = ConversationState.from_dict(future_data)
except SchemaVersionError as e:
    print(f"Cannot load: {e}")
    # "Cannot downgrade from schema 2.0.0 to 1.0.0"
```

**Solution**: Upgrade your `dataknobs_llm` package to support the newer schema.

### Unknown Minor/Patch Versions

Loading a conversation with an unknown minor or patch version logs a warning but attempts to proceed:

```python
# Current version: 1.0.0
# Loading conversation with version: 1.2.0

# Logs warning: "No migration path defined from 1.2.0 to 1.0.0. Using data as-is."
# Attempts to load - may work if changes are backward compatible
```

## Best Practices

### When to Increment Version

**MAJOR** (1.0.0 → 2.0.0):
- Remove fields
- Rename fields
- Change field types
- Restructure data format
- Any change that breaks backward compatibility

**MINOR** (1.0.0 → 1.1.0):
- Add optional fields with defaults
- Add new node types
- Extend existing fields (e.g., add new metadata keys)
- Backward-compatible additions

**PATCH** (1.0.0 → 1.0.1):
- Bug fixes in migration logic
- Documentation updates
- No schema changes

### Migration Testing

Always test migrations thoroughly:

```python
# Test forward migration
def test_migration_1_0_to_1_1():
    old_data = create_1_0_data()
    state = ConversationState.from_dict(old_data)
    assert state.schema_version == "1.1.0"
    validate_1_1_structure(state)

# Test roundtrip
def test_migration_roundtrip():
    old_data = create_1_0_data()
    state = ConversationState.from_dict(old_data)
    new_data = state.to_dict()
    state2 = ConversationState.from_dict(new_data)
    assert_states_equal(state, state2)

# Test multiple versions
def test_migration_chain():
    v0_data = create_0_0_data()
    state = ConversationState.from_dict(v0_data)
    # Should migrate through all intermediate versions
    assert state.schema_version == SCHEMA_VERSION
```

### Production Considerations

1. **Test migrations on copies** before upgrading production data
2. **Monitor migration logs** to track which conversations are being migrated
3. **Backup data** before deploying schema changes
4. **Plan rollback strategy** for major version changes
5. **Communicate breaking changes** to users in advance

### Versioning Guidelines

1. **Be conservative with major versions** - they require user action
2. **Prefer minor versions** for new features when possible
3. **Document all changes** in version history
4. **Test migration paths** from all previous versions
5. **Keep migrations simple** - complex transformations are error-prone

## Monitoring

Track schema migrations in production:

```python
import logging

# Set up logging to track migrations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Migrations will log:
# "Migrating conversation conv-123 from schema 0.0.0 to 1.0.0"

# Monitor these logs to understand:
# - How many old conversations exist
# - Which versions are being migrated from
# - Migration performance
```

## FAQ

### Q: What happens to conversations created before versioning?

**A**: They are treated as version "0.0.0" and automatically migrated to the current version (1.0.0) when loaded. The migration is seamless and requires no user action.

### Q: Can I query the schema version of a stored conversation?

**A**: Yes, the `schema_version` field is stored in the serialized data. You can check it before loading:

```python
# Load raw data
record = await backend.read(conversation_id)
data = {field.name: field.value for field in record.fields.values()}

# Check version
version = data.get("schema_version", "0.0.0")
print(f"Conversation uses schema version: {version}")
```

### Q: What if migration fails?

**A**: Migration errors raise `SchemaVersionError` with details. The original data is not modified. You can:
1. Check the error message
2. Report the issue
3. Try loading with an older version of the package
4. Restore from backup if needed

### Q: Can I disable automatic migration?

**A**: No, migration is automatic and required for consistency. However, you can:
- Load the raw data without deserializing
- Check the version first
- Handle migration errors appropriately

### Q: How do I handle a production schema upgrade?

**A**:
1. Test migration on a copy of production data
2. Plan a maintenance window if needed (for major versions)
3. Deploy new version
4. Monitor migration logs
5. Conversations migrate on first load (lazy migration)
6. Keep backups until confirmed stable

## Summary

Schema versioning provides:
- **Backward compatibility** for old conversations
- **Automatic migration** on load
- **Protection against** downgrades
- **Clear upgrade path** for schema changes
- **Monitoring** via migration logs

The system is designed to be **transparent** to users while providing **robust data migration** capabilities for long-term evolution of the conversation storage format.

---

**For questions or issues**: Report at https://github.com/kbs-labs/dataknobs/issues

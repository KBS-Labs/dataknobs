---
globs:
  - "docs/**/*.md"
  - "packages/*/docs/**/*.md"
  - "**/README.md"
---

# Dual Documentation System

Dataknobs maintains TWO documentation locations that must stay synchronized.

## Documentation Locations

### 1. Package-Specific Docs
Location: `packages/<PACKAGE>/docs/`

Purpose:
- Package README and overview
- API reference specific to the package
- Package-specific examples
- Lives with the code it documents

### 2. MkDocs Site Docs
Location: `docs/packages/<PACKAGE>/`

Purpose:
- Unified documentation site
- Cross-package navigation
- Getting started guides
- Rendered by MkDocs

## Update Requirements

When updating documentation for any package:

1. **Update BOTH locations:**
   - `packages/<PACKAGE>/docs/` - Package-specific docs
   - `docs/packages/<PACKAGE>/` - MkDocs site docs

2. **Verify MkDocs build succeeds:**
   ```bash
   uv run mkdocs build --strict
   ```
   - Must complete without errors
   - Warnings about missing files or broken links must be fixed

3. **Check navigation:**
   - Ensure new pages are added to `mkdocs.yml` if needed
   - Verify cross-links work

## Package-to-MkDocs Mapping

| Package | Package Docs | MkDocs Docs |
|---------|--------------|-------------|
| common | packages/common/docs/ | docs/packages/common/ |
| config | packages/config/docs/ | docs/packages/config/ |
| data | packages/data/docs/ | docs/packages/data/ |
| llm | packages/llm/docs/ | docs/packages/llm/ |
| bots | packages/bots/docs/ | docs/packages/bots/ |
| fsm | packages/fsm/docs/ | docs/packages/fsm/ |
| structures | packages/structures/docs/ | docs/packages/structures/ |
| utils | packages/utils/docs/ | docs/packages/utils/ |
| xization | packages/xization/docs/ | docs/packages/xization/ |

## MkDocs Validation

After any documentation change:

```bash
# Build with strict mode to catch warnings as errors
uv run mkdocs build --strict

# Preview locally
uv run mkdocs serve
```

Common issues to check:
- Missing nav entries in mkdocs.yml
- Broken internal links
- Missing images or assets
- Orphaned pages not in navigation

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

4. **Classify the pair in the doc-mirror manifest:**
   ```bash
   python3 bin/docs-mirror-check.py
   ```
   Docs in scope must be classified in
   `.dataknobs/docs-mirror-manifest.json`, and an unclassified one fails
   the guard. `mkdocs build --strict` will NOT catch that — run the guard
   too (or `bin/run-quality-checks.sh`, which invokes it).

   **Scope is per package, and not yet complete everywhere.** A package
   with `"recursive": true` requires every `*.md` at any depth to be
   classified, so a new doc there fails until you do. A package without it
   requires only *top-level* `*.md` — a new doc under a subdirectory (every
   bots guide, for instance) passes unclassified and gets no verification
   at all.

   | Scope | Packages |
   |---|---|
   | `recursive: true` — a new doc at any depth is caught | common, config, structures, utils |
   | top-level only — a new nested doc is **not** caught | bots, data, fsm, llm, xization |

   So classify a new nested doc in one of the right-hand packages even
   though nothing forces you to. Opting a package in means classifying
   everything already nested there, which is why it is being done one
   package at a time.

   Pick the class by how the two copies are kept in agreement:

   | Class | Use when |
   |---|---|
   | `transclude` | Site page is a `--8<--` include of the package source (**preferred for new docs** — drift is structurally impossible) |
   | `symlink` | Site page symlinks the source; no intra-doc links need site-form rewriting |
   | `mirror` | Hand-authored copy, content-guarded; `--fix` regenerates it |
   | `diverge` | Intentional divergence; recorded, not content-checked |
   | `package_only` / `site_only` | Genuinely unpaired — **not** a fallback for a pair you did not want to classify |

   A paired entry may point at a subdirectory on either side (a package
   source under `guides/`, or a site page under `guides/` as every bots
   guide is), so prefer a real pair over `package_only`/`site_only`:
   an unpaired classification opts the file out of per-class verification.

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

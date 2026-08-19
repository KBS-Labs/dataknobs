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
   at all. It can still be classified individually, in any class; the
   scope decides what the guard *demands*, not what it accepts.

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
   | `diverge` | Intentional divergence; recorded, not content-checked. Add `shared_sections` for any block that must still stay identical |
   | `package_only` / `site_only` | Genuinely unpaired — **not** a fallback for a pair you did not want to classify |

   **Any** entry may point at a subdirectory on either side (a package
   source under `guides/`, or a site page under `guides/` as every bots
   guide is) — paired or unpaired, whatever the package's scope. So
   prefer a real pair over `package_only`/`site_only`: those two classes
   verify only that the file exists and that nothing in the other tree
   pairs with it, never what the file contains.

   That preference is now enforced rather than advisory. The guard fails
   an unpaired entry whose counterpart exists in the other tree, matched
   on the canonicalized basename at any depth — because a real pair
   recorded as unpaired gets no content check while still reporting
   green. If two documents share a name but are genuinely
   different, that is a `diverge` with a reason, not a `package_only`.

   The subdirectory rule reached the unpaired classes later than the
   paired ones, and the gap was not cosmetic: a genuinely unpaired
   nested page could not be classified at all under a top-level scope,
   so the only ways to record it were to opt the whole package into
   `recursive: true` or to leave it unverified. Both classes accept a
   nested path now, which is the cheap way to cover one nested doc
   without reconciling the package's whole tree.

   **Two documents that differ overall but share one block.** Neither
   `mirror` (whole file must match) nor `transclude` (whole file is an
   include) can express this, so the block gets hand-copied into both
   sides and is verified by nothing. Use `diverge` with `shared_sections`:
   wrap the block in the package source with pymdownx section markers
   inside HTML comments, so they stay invisible when the doc is read on
   GitHub —

   ```markdown
   <!-- --8<-- [start:catching-api-errors] -->
   #### Catching these errors
   ...
   <!-- --8<-- [end:catching-api-errors] -->
   ```

   — and pull it into the site page where the copy used to be:

   ```markdown
   --8<-- "packages/bots/docs/MULTI_TENANT.md:catching-api-errors"
   ```

   Then declare it, so replacing the include with a fresh copy fails the
   guard rather than silently restoring the drift:

   ```json
   { "package": "MULTI_TENANT.md", "site": "guides/bot-manager.md",
     "reason": "...", "shared_sections": ["catching-api-errors"] }
   ```

   Deleting a marker breaks `mkdocs build --strict` too — a missing
   section raises `SnippetMissingError` rather than including nothing.

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

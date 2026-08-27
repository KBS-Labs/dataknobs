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

## One Document, One Name

Both trees spell a document the same way: **lower-hyphen** —
`user-guide.md`, `configuration.md`, `05.updated-plan.md`. Never
`USER_GUIDE.md`, never `template_vars.md`.

This is not a style preference. A `symlink` or `transclude` page is one
file served at two paths, so a link inside it is read from both — and
`[Configuration](configuration.md)` can be correct in both trees only if
both trees agree on the filename. While they did not, **89 package-tree
links were broken** and the rendered site was clean, so nothing reported
it for as long as the guard had existed.

`bin/docs-mirror-check.py` enforces two halves of that:

| Check | What fails |
|---|---|
| **doc spelling** | a package doc whose filename is not lower-hyphen. `README.md` is the one exemption — GitHub renders it as a directory index, and `readme.md` on a case-sensitive host does not get that treatment. |
| **name parity** | a `symlink` / `transclude` pair spelled differently on its two sides. `diverge` is exempt: that class records two genuinely *different* documents, so requiring a shared name would contradict it. |

The site tree is covered through the pair rather than directly, so a
genuinely site-only page may keep a name taken from the module it
documents (`docs/packages/fsm/api/async_simple.md`). Served from one tree,
it has no second tree for its link text to disagree with.

Only the basename is compared. The two trees nest some documents
differently — `packages/llm/docs/best-practices.md` against
`docs/packages/llm/guides/best-practices.md` — and no filename reconciles
that.

### Links must resolve, in every tree the doc is served from

Relative `.md` links are resolved case-**sensitively**, because
`Path.exists()` is not: on macOS it answers *yes* for `configuration.md`
when only `CONFIGURATION.md` is on disk. That is how 31 of the 89 stayed
invisible on the machines they were written on.

**Every relative `.md` link must resolve, and the guard fails when one
does not.** Two shapes, one verdict, different remedies — which is why
the messages differ:

| Shape | What it means | Remedy the message names |
|---|---|---|
| **spelling** | the document is in that directory under another name | rename the file, or correct the link |
| **absent under any spelling** | the trees nest it differently, or it is site-native, or it is gone | absolute site URL, publish the target, or name it in prose |

The second half was **printed and counted rather than failed** while the
question it posed was open: what may a package doc link to, when no
relative path can reach the target from both trees? A cross-package link
cannot — the package tree carries a `/docs/` segment the site tree does
not, so `../../bots/docs/x.md` and `../../bots/x.md` cannot both be
right. Aligning the nesting was priced at 58 published URLs to fix one
link, and rejected on that number.

So the answer is per shape, and **none of it is declared anywhere**:

- **Absolute site URL** — `https://kbs-labs.github.io/dataknobs/packages/<pkg>/<page>/`.
  The default for a cross-package target, a site-native one, or a
  one-file pair whose two directories disagree. Derive it from the
  **site file's path on disk**, not from the nav: `use_directory_urls`
  defaults to true, so `docs/packages/bots/guides/tools.md` is
  `/packages/bots/guides/tools/` — the `guides/` segment is part of the
  URL. The cost is real and accepted: `--strict` no longer validates the
  target, and `mkdocs serve` sends a local preview to production.
- **Publish the target** into the package tree and symlink the site page,
  when the target is package documentation that happens to live only on
  the site. Root-cause fix, changes no URL — but check the target's *own*
  onward links first, or the move trades two findings for two more.
- **Name it in prose, without a link** — `wizard-subflows.md` carries the
  precedent and its reason in an HTML comment.
- **Repair the relative path**, when the document is served from one tree
  only (`package_only`, `site_only`) or the pair is two independent files
  (`diverge`). No cross-tree constraint exists there; such a link is
  simply wrong.

Nothing records which was chosen, and nothing needs to: an absolute URL
is not a relative `.md` link and a prose mention is not a link at all, so
all of them are invisible to the guard by construction rather than by
allowlist. That is what keeps the standing cost of this rule at zero.

## Update Requirements

When updating documentation for any package:

1. **Name the doc `lower-hyphen.md`, identically in both trees** — see
   [One Document, One Name](#one-document-one-name) above.

2. **Update BOTH locations:**
   - `packages/<PACKAGE>/docs/` - Package-specific docs
   - `docs/packages/<PACKAGE>/` - MkDocs site docs

3. **Verify MkDocs build succeeds:**
   ```bash
   uv run mkdocs build --strict
   ```
   - Must complete without errors
   - Warnings about missing files or broken links must be fixed

4. **Check navigation:**
   - Ensure new pages are added to `mkdocs.yml` if needed
   - Verify cross-links work

5. **Classify the pair in the doc-mirror manifest:**
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
   | `transclude` | Site page is a `--8<--` include of the package source, **and nothing else** — no heading of its own. The default for a new pair |
   | `symlink` | Site page symlinks the source. Same guarantee; use it where a symlink is what is already there |
   | `diverge` | Intentional divergence; recorded, not content-checked. Add `shared_sections` for any block that must still stay identical |
   | `package_only` / `site_only` | Genuinely unpaired — **not** a fallback for a pair you did not want to classify |

   **Do not put a title above the include.** The source's own H1 renders as
   the page title; a second one above it renders a duplicate `<h1>`. That
   was true of 24 of the 31 transclude pages and of none of the other 7,
   with no exception in either direction, so the bare form is not a style
   preference — it is the one that renders correctly.

   **There was a sixth class, `mirror`,** and it is worth knowing why it is
   gone rather than finding its name in an old commit. It held a
   hand-authored site copy kept byte-identical to its package source by a
   content comparison, with a `line_exceptions` list for the lines that had
   to read differently in each tree. That list was its whole reason to
   exist: it was the only class holding two real files, so it was the only
   one that could carry a per-tree link text. Once every such link became
   an absolute site URL there was nothing left to express, and a class that
   guarantees *by comparison* what two others guarantee *by construction*
   is the weaker way to say the same thing. A key the guard does not know —
   `mirror` today, a typo any day — is now refused rather than skipped,
   because an entry under an unrecognised key classified nothing and left
   both files silently unverified.

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

   **Two documents that differ overall but share one block.** A whole-file
   `transclude` cannot express this, so the block gets hand-copied into both
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
   --8<-- "packages/bots/docs/multi-tenant.md:catching-api-errors"
   ```

   Then declare it, so replacing the include with a fresh copy fails the
   guard rather than silently restoring the drift:

   ```json
   { "package": "multi-tenant.md", "site": "guides/bot-manager.md",
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

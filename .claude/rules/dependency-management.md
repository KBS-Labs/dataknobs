# Dependency Management

## Open-Source Selection Criteria

Before adding any new dependency, ALL criteria must be met:

| Criterion | Requirement |
|-----------|-------------|
| **Maturity** | 1.0+ release or 2+ years of production use |
| **Maintenance** | Active commits within the last 6 months |
| **Adoption** | Widely used in the ecosystem |
| **License** | Permissive only (see below) |
| **Scope** | No excessive transitive dependencies |

**Prefer libraries already in the dependency tree** before adding new ones. Check `pyproject.toml` and `uv.lock` first.

Since dataknobs is infrastructure consumed by many projects, every dependency we add becomes a transitive dependency for all consumers. Be conservative.

## License Rules (Non-Negotiable)

This is especially critical for dataknobs — as a library, our dependency licenses propagate to every consuming project.

### Permitted Licenses

MIT, Apache 2.0, BSD (2-clause and 3-clause), ISC, Unlicense, CC0

### Prohibited Licenses

Any copyleft license that would require open-sourcing proprietary downstream code:
- GPL (all versions), AGPL, SSPL, EUPL
- LGPL (when statically linked or boundary is unclear)
- MPL 2.0 (unless strictly file-scoped and verified)

### Rules

- **MIT is the gold standard.** If a library is not MIT-licensed, verify its license explicitly before adopting.
- **Transitive dependencies matter.** A permissive library that depends on a GPL library still creates a GPL obligation. Check the full dependency tree.
- **When no permissive alternative exists:** Build it ourselves (in the appropriate dataknobs package) or enhance an existing dataknobs construct. Do not adopt the copyleft dependency.

## Upgrade Over Duplicate

When existing code (in dataknobs or dependencies) handles ~80% of the use case:

- **Extend it** with a parameter or optional behavior — don't create `_v2` variants
- **Add to the existing module** — don't create a parallel one in a new file
- Ensure backward compatibility and test both old and new behavior

This applies within dataknobs packages too: if `dataknobs-utils` has an HTTP helper that's close to what you need, enhance it rather than building a new one in `dataknobs-llm`.

## Anti-Patterns

Stop and revisit the reuse hierarchy if you find yourself:

- **Copy-pasting** from another module and tweaking — extract a shared function
- **Creating `_v2` / `_new` variants** — upgrade the original
- **Wrapping** a library call with no added value — call it directly
- **Duplicating boilerplate** across packages — extract to `dataknobs-common` or `dataknobs-utils`
- **Re-implementing stdlib** — `datetime`, `pathlib`, `itertools`, `functools`, `collections` cover more than you think
- **Adding a dependency** for something a current dependency already handles
- **Duplicating across dataknobs packages** — if two packages need the same utility, it belongs in `dataknobs-common` or `dataknobs-utils`

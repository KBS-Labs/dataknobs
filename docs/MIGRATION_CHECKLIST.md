# Dataknobs Migration Checklist

This checklist guides the migration of Dataknobs from a single Poetry-managed package to a monorepo with modular packages using `uv`. Check off tasks as they are completed to track progress.

## Planning
- [ ] Identify logical components (e.g., `tool-a`, `tool-b`) in the current `myproject` package.
- [ ] Determine if shared code requires a `myproject-common` package.
- [ ] Define versioning strategy (e.g., start new packages at `1.0.0`).
- [ ] Plan deprecation timeline for the `legacy` package (e.g., 6–12 months).

## Monorepo Setup
- [ ] Create monorepo directory structure:
  - `packages/legacy/`, `packages/tool-a/`, `packages/tool-b/`, `packages/common/` (if needed).
  - `docs/`, `.github/workflows/`, `README.md`, `LICENSE`.
- [ ] Initialize new packages with `uv init` in each `packages/*/` directory.
- [ ] Create optional `uv.toml` for monorepo-wide settings.

## Migrate to `uv`
- [ ] Install `uv` (`pip install uv`).
- [ ] Convert Poetry’s `pyproject.toml` to `uv`-compatible format for each package:
  - `legacy` package (`myproject`).
  - `tool-a` package (`myproject-tool-a`).
  - `tool-b` package (`myproject-tool-b`).
  - `common` package (`myproject-common`, if needed).
- [ ] Migrate dependencies using `uv add <dependency>`.
- [ ] Generate `uv.lock` files with `uv lock` in each package directory.

## Code Refactoring
- [ ] Move code to new packages (`tool-a`, `tool-b`, `common`).
- [ ] Update imports to reflect new structure (e.g., `from myproject_tool_a import func`).
- [ ] Configure `legacy` package to re-export APIs from new packages.
- [ ] Extract shared code to `myproject-common` (if applicable).
- [ ] Update `legacy` package’s `pyproject.toml` to depend on new packages (e.g., `myproject-tool-a==1.0.0`).

## Testing
- [ ] Write unit tests for each package in `packages/*/tests/`.
- [ ] Write integration tests to verify interoperability between `tool-a`, `tool-b`, and `legacy`.
- [ ] Run tests locally with `uv run pytest`.
- [ ] Test local installation (`uv build`, `pip install dist/myproject-*.whl`).

## CI/CD
- [ ] Create GitHub Actions workflow (`ci.yml`) for testing across Python 3.8–3.12.
- [ ] Create release workflow (`release.yml`) for publishing to PyPI using `uv publish`.
- [ ] Test CI pipelines on a feature branch.

## Documentation
- [ ] Update `README.md` with new structure and installation instructions.
- [ ] Add migration guide to `docs/` (e.g., “Upgrading to the Monorepo”).
- [ ] Add deprecation notice to `legacy` package’s `README.md` and docstrings.
- [ ] Set up `mkdocs` for unified documentation in `docs/`.
- [ ] Add PyPI, CI, and coverage badges to each package’s `README.md`.

## Release
- [ ] Publish pre-release versions to TestPyPI (`uv publish --index https://test.pypi.org/legacy/`).
- [ ] Solicit feedback from active users on pre-releases.
- [ ] Tag releases (e.g., `git tag tool-a/v1.0.0`) for each package.
- [ ] Publish new packages (`myproject-tool-a`, `myproject-tool-b`, `myproject-common`) to PyPI.
- [ ] Publish updated `legacy` package (`myproject`) to PyPI.
- [ ] Create release notes detailing the migration and deprecation plan.

## Communication
- [ ] Announce migration in GitHub release notes.
- [ ] Update PyPI project description with migration details.
- [ ] Post announcements to community channels (e.g., X, mailing lists).
- [ ] Monitor GitHub issues for user feedback.

## Post-Migration
- [ ] Verify all packages install and work correctly (`pip install myproject`, `pip install myproject-tool-a`).
- [ ] Respond to user issues or questions promptly.
- [ ] Plan follow-up releases for bug fixes or improvements.
- [ ] Schedule deprecation of `legacy` package (e.g., v2.0.0).

## Notes
- Use `black`, `isort`, and `mypy` for code quality (`uv run black .`).
- Pin exact versions in `legacy` package to avoid conflicts.
- Keep the old Poetry-based repository as a branch or fork until the transition is stable.

# Dataknobs Migration Checklist

This checklist guides the migration of Dataknobs from a single Poetry-managed package to a monorepo with modular packages using `uv`. Check off tasks as they are completed to track progress.

## Planning
- [x] Identify logical components (structures, utils, xization) in the current `dataknobs` package.
- [x] Determine if shared code requires a `dataknobs-common` package.
- [x] Define versioning strategy (start new packages at `1.0.0`).
- [ ] Plan deprecation timeline for the `legacy` package (e.g., 6–12 months).

## Monorepo Setup
- [x] Create monorepo directory structure:
  - `packages/legacy/`, `packages/structures/`, `packages/utils/`, `packages/xization/`, `packages/common/`.
  - `docs/`, `.github/workflows/`, `README.md`, `LICENSE`.
- [x] Initialize new packages with `uv init` in each `packages/*/` directory.
- [ ] Create optional `uv.toml` for monorepo-wide settings.

## Migrate to `uv`
- [x] Install `uv` (already installed via Homebrew).
- [x] Convert Poetry's `pyproject.toml` to `uv`-compatible format for each package:
  - `legacy` package (`dataknobs`).
  - `structures` package (`dataknobs-structures`).
  - `utils` package (`dataknobs-utils`).
  - `xization` package (`dataknobs-xization`).
  - `common` package (`dataknobs-common`).
- [x] Migrate dependencies via direct pyproject.toml editing.
- [x] Generate `uv.lock` file with `uv lock` at workspace level.

## Code Refactoring
- [x] Move code to new packages (`structures`, `utils`, `xization`, `common`).
- [ ] Update imports to reflect new structure (e.g., `from dataknobs_structures import func`).
- [x] Configure `legacy` package to re-export APIs from new packages.
- [ ] Extract shared code to `dataknobs-common` (if applicable).
- [x] Update `legacy` package's `pyproject.toml` to depend on new packages (e.g., `dataknobs-structures==1.0.0`).

## Testing
- [x] Move existing unit tests to each package in `packages/*/tests/`.
- [x] Update test imports to use new package names.
- [x] Fix test resource paths for monorepo structure.
- [ ] Write integration tests to verify interoperability between packages.
- [x] Run tests locally with `uv run pytest` (all 94 tests passing).
- [ ] Test local installation (`uv build`, `pip install dist/dataknobs-*.whl`).

## CI/CD
- [x] Create GitHub Actions workflow (`ci.yml`) for testing across Python 3.10–3.13.
- [x] Create release workflow (`release.yml`) for publishing to PyPI using `uv publish`.
- [x] Create dependency update workflow for automated updates.
- [ ] Test CI pipelines on a feature branch.

## Docker & Infrastructure
- [x] Update Dockerfiles to use `uv` instead of Poetry.
- [x] Create docker-compose.yml for easy development setup.
- [x] Update tox.ini to use `uv` commands.
- [x] Create .dockerignore file.
- [x] Add docker/README.md with usage instructions.

## Documentation
- [x] Update root `README.md` with new structure and installation instructions.
- [x] Add migration guide to `docs/` (MIGRATION_GUIDE.md).
- [x] Add deprecation notice to `legacy` package's `__init__.py`.
- [x] Add deprecation notice to `legacy` package's `README.md`.
- [ ] Set up `mkdocs` for unified documentation in `docs/`.
- [x] Add README.md to all packages (structures, utils, xization, common, legacy).

## Release
- [ ] Publish pre-release versions to TestPyPI (`uv publish --index https://test.pypi.org/legacy/`).
- [ ] Solicit feedback from active users on pre-releases.
- [ ] Tag releases (e.g., `git tag structures/v1.0.0`) for each package.
- [ ] Publish new packages (`dataknobs-structures`, `dataknobs-utils`, `dataknobs-xization`, `dataknobs-common`) to PyPI.
- [ ] Publish updated `legacy` package (`dataknobs`) to PyPI.
- [ ] Create release notes detailing the migration and deprecation plan.

## Communication
- [ ] Announce migration in GitHub release notes.
- [ ] Update PyPI project description with migration details.
- [ ] Post announcements to community channels (e.g., X, mailing lists).
- [ ] Monitor GitHub issues for user feedback.

## Post-Migration
- [ ] Verify all packages install and work correctly (`pip install dataknobs`, `pip install dataknobs-structures`).
- [ ] Respond to user issues or questions promptly.
- [ ] Plan follow-up releases for bug fixes or improvements.
- [ ] Schedule deprecation of `legacy` package (e.g., v2.0.0).

## Notes
- Use `black`, `isort`, and `mypy` for code quality (`uv run black .`).
- Pin exact versions in `legacy` package to avoid conflicts.
- Keep the old Poetry-based repository as a branch or fork until the transition is stable.

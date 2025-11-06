#!/usr/bin/env python3
"""Create a new DataKnobs package with complete ecosystem integration.

This script automates the creation of a new package and its integration into
the DataKnobs monorepo, including:

- Package structure creation
- Package registry update (.dataknobs/packages.json)
- GitHub workflow updates (docs.yml, quality-validation.yml, release.yml)
- README.md integration
- Root pyproject.toml integration
- MkDocs configuration (if docs are enabled)
- Package index documentation

Usage:
    ./bin/create-package.py <package-name> [options]

Examples:
    # Create a new core package with documentation
    ./bin/create-package.py ml --description "Machine learning utilities" --version 0.1.0

    # Create an experimental package without docs
    ./bin/create-package.py experimental-feature --category experimental --no-docs

    # Dry run to see what would be changed
    ./bin/create-package.py mypackage --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class PackageCreator:
    """Handles creation and integration of new DataKnobs packages."""

    def __init__(self, repo_root: Path, dry_run: bool = False):
        self.repo_root = repo_root
        self.dry_run = dry_run
        self.changes: list[str] = []

    def log_change(self, message: str) -> None:
        """Log a change that was or would be made."""
        prefix = "[DRY RUN] " if self.dry_run else ""
        print(f"{prefix}{message}")
        self.changes.append(message)

    def create_package_structure(
        self,
        name: str,
        description: str,
        version: str = "0.1.0"
    ) -> bool:
        """Create the basic package directory structure.

        Args:
            name: Package name (e.g., 'bots')
            description: Short package description
            version: Initial version number

        Returns:
            True if structure was created, False if it already exists
        """
        package_dir = self.repo_root / "packages" / name

        if package_dir.exists():
            print(f"❌ Package directory already exists: {package_dir}")
            return False

        # Define directory structure
        dirs_to_create = [
            package_dir / "src" / f"dataknobs_{name}",
            package_dir / "tests",
            package_dir / "docs",
        ]

        files_to_create = {
            package_dir / "pyproject.toml": self._generate_pyproject_toml(name, description, version),
            package_dir / "src" / f"dataknobs_{name}" / "__init__.py": self._generate_init_py(name, version),
            package_dir / "README.md": self._generate_package_readme(name, description),
            package_dir / "tests" / "__init__.py": "",
            package_dir / "tests" / f"test_{name}.py": self._generate_test_file(name),
        }

        if not self.dry_run:
            # Create directories
            for dir_path in dirs_to_create:
                dir_path.mkdir(parents=True, exist_ok=True)

            # Create files
            for file_path, content in files_to_create.items():
                file_path.write_text(content)

        for dir_path in dirs_to_create:
            self.log_change(f"✅ Created directory: {dir_path.relative_to(self.repo_root)}")

        for file_path in files_to_create:
            self.log_change(f"✅ Created file: {file_path.relative_to(self.repo_root)}")

        return True

    def update_package_registry(
        self,
        name: str,
        pypi_name: str,
        description: str,
        version: str,
        category: str,
        requires_docs_build: bool
    ) -> None:
        """Update .dataknobs/packages.json with new package."""
        registry_path = self.repo_root / ".dataknobs" / "packages.json"

        with open(registry_path) as f:
            registry = json.load(f)

        # Check if package already exists
        existing = [p for p in registry["packages"] if p["name"] == name]
        if existing:
            print(f"⚠️  Package '{name}' already exists in registry")
            return

        # Add new package entry
        new_package = {
            "name": name,
            "pypi_name": pypi_name,
            "description": description,
            "version": version,
            "category": category,
            "requires_docs_build": requires_docs_build,
            "deprecated": False
        }

        registry["packages"].append(new_package)

        # Sort packages by category and name
        def sort_key(pkg: dict[str, Any]) -> tuple[int, str]:
            category_order = {"core": 0, "experimental": 1, "legacy": 2}
            return (category_order.get(pkg["category"], 99), pkg["name"])

        registry["packages"].sort(key=sort_key)

        if not self.dry_run:
            with open(registry_path, "w") as f:
                json.dump(registry, f, indent=2)
                f.write("\n")  # Add trailing newline

        self.log_change(f"✅ Added '{name}' to .dataknobs/packages.json")

    def update_workflow_docs(self, name: str) -> None:
        """Update GitHub workflow files for documentation builds."""
        workflow_files = [
            ".github/workflows/docs.yml",
            ".github/workflows/quality-validation.yml"
        ]

        for workflow_file in workflow_files:
            workflow_path = self.repo_root / workflow_file
            if not workflow_path.exists():
                print(f"⚠️  Workflow file not found: {workflow_file}")
                continue

            content = workflow_path.read_text()

            # Find the last "uv pip install --system -e packages/" line
            lines = content.split("\n")
            insert_index = None

            for i in range(len(lines) - 1, -1, -1):
                if "uv pip install --system -e packages/" in lines[i]:
                    insert_index = i + 1
                    break

            if insert_index is None:
                print(f"⚠️  Could not find install location in {workflow_file}")
                continue

            new_line = f"          uv pip install --system -e packages/{name}"

            # Check if already present
            if new_line.strip() in content:
                print(f"⚠️  Package '{name}' already in {workflow_file}")
                continue

            lines.insert(insert_index, new_line)
            new_content = "\n".join(lines)

            if not self.dry_run:
                workflow_path.write_text(new_content)

            self.log_change(f"✅ Updated {workflow_file}")

    def update_release_workflow(self, name: str, category: str) -> None:
        """Update release workflow with new package choice."""
        if category == "experimental":
            print(f"ℹ️  Skipping release workflow for experimental package '{name}'")
            return

        workflow_path = self.repo_root / ".github/workflows/release.yml"
        if not workflow_path.exists():
            print("⚠️  Release workflow not found")
            return

        content = workflow_path.read_text()
        lines = content.split("\n")

        # Find the options section
        options_start = None
        all_index = None

        for i, line in enumerate(lines):
            if "options:" in line:
                options_start = i
            if options_start and "- all" in line:
                all_index = i
                break

        if options_start is None or all_index is None:
            print("⚠️  Could not find options section in release.yml")
            return

        # Insert before "- all"
        new_option = f"          - {name}"

        if new_option.strip() in content:
            print(f"⚠️  Package '{name}' already in release.yml")
            return

        lines.insert(all_index, new_option)

        # Also update the description line if it exists
        for i, line in enumerate(lines):
            if "description:" in line and "Package to release" in line:
                # Add to the example list
                if name not in line:
                    lines[i] = line.rstrip(")") + f", {name})"
                break

        new_content = "\n".join(lines)

        if not self.dry_run:
            workflow_path.write_text(new_content)

        self.log_change("✅ Updated .github/workflows/release.yml")

    def update_readme(self, name: str, pypi_name: str, description: str, category: str) -> None:
        """Update README.md with new package."""
        if category != "core":
            print(f"ℹ️  Skipping README for non-core package '{name}'")
            return

        readme_path = self.repo_root / "README.md"
        content = readme_path.read_text()
        lines = content.split("\n")

        # Find package list section
        packages_section_start = None
        for i, line in enumerate(lines):
            if "## 📦 Packages" in line or "## Packages" in line:
                packages_section_start = i
                break

        if packages_section_start is None:
            print("⚠️  Could not find packages section in README.md")
            return

        # Find where to insert (before structures, utils, or common/legacy)
        insert_index = None
        for i in range(packages_section_start, len(lines)):
            if "- **[dataknobs-structures]" in lines[i]:
                insert_index = i
                break
            if "- **[dataknobs-utils]" in lines[i]:
                insert_index = i
                break
            if "- **[dataknobs-common]" in lines[i] or "- **[dataknobs]" in lines[i]:
                insert_index = i
                break

        if insert_index is None:
            print("⚠️  Could not find insertion point in README.md")
            return

        new_line = f"- **[{pypi_name}](packages/{name}/)**: {description}"

        if pypi_name in content:
            print(f"⚠️  Package '{pypi_name}' already in README.md")
            return

        lines.insert(insert_index, new_line)

        # Also add to installation section
        install_section_found = False
        for i, line in enumerate(lines):
            if "# Install specific packages" in line:
                # Find the end of pip install commands
                j = i + 1
                while j < len(lines) and lines[j].startswith("pip install dataknobs-"):
                    j += 1
                # Insert before the blank line or comment
                new_install = f"pip install {pypi_name}"
                if new_install not in content:
                    lines.insert(j, new_install)
                    install_section_found = True
                break

        new_content = "\n".join(lines)

        if not self.dry_run:
            readme_path.write_text(new_content)

        self.log_change("✅ Updated README.md")

    def update_pyproject_toml(self, name: str, pypi_name: str) -> None:
        """Update root pyproject.toml with new package."""
        pyproject_path = self.repo_root / "pyproject.toml"
        content = pyproject_path.read_text()
        lines = content.split("\n")

        updates_needed = {
            "dependencies": False,
            "sources": False,
            "mypy_path": False,
            "known_first_party": False,
        }

        # 1. Add to dependencies
        for i, line in enumerate(lines):
            if line.strip() == "dependencies = [":
                # Find the end of dependencies
                j = i + 1
                while j < len(lines) and not lines[j].strip() == "]":
                    j += 1
                # Insert before the last item or closing bracket
                new_dep = f'    "{pypi_name}",'
                if pypi_name not in content:
                    lines.insert(j, new_dep)
                    updates_needed["dependencies"] = True
                break

        # 2. Add to [tool.uv.sources]
        for i, line in enumerate(lines):
            if line.strip() == "[tool.uv.sources]":
                # Find the last workspace source
                j = i + 1
                while j < len(lines) and "workspace = true" in lines[j]:
                    j += 1
                new_source = f'{pypi_name} = {{ workspace = true }}'
                if pypi_name not in content:
                    lines.insert(j, new_source)
                    updates_needed["sources"] = True
                break

        # 3. Add to mypy_path
        for i, line in enumerate(lines):
            if line.startswith("mypy_path = "):
                if f"packages/{name}/src" not in line:
                    # Add to the path
                    path_parts = line.split('"')[1].split(":")
                    path_parts.append(f"packages/{name}/src")
                    new_path = ":".join(path_parts)
                    lines[i] = f'mypy_path = "{new_path}"'
                    updates_needed["mypy_path"] = True
                break

        # 4. Add to known-first-party
        for i, line in enumerate(lines):
            if "known-first-party = [" in line:
                if f"dataknobs_{name}" not in line:
                    # Extract the list and add new package
                    start = line.index("[")
                    end = line.rindex("]")
                    packages_str = line[start + 1:end]
                    packages = [p.strip().strip('"') for p in packages_str.split(",") if p.strip()]
                    packages.append(f"dataknobs_{name}")
                    packages.sort()
                    new_packages_str = ", ".join(f'"{p}"' for p in packages)
                    lines[i] = f'known-first-party = [{new_packages_str}]'
                    updates_needed["known_first_party"] = True
                break

        if any(updates_needed.values()):
            new_content = "\n".join(lines)
            if not self.dry_run:
                pyproject_path.write_text(new_content)
            self.log_change("✅ Updated root pyproject.toml")
        else:
            print("ℹ️  Package already in pyproject.toml")

    def create_package(
        self,
        name: str,
        description: str,
        version: str = "0.1.0",
        category: str = "core",
        requires_docs_build: bool = True
    ) -> bool:
        """Create a new package and integrate it into the ecosystem.

        Args:
            name: Package name (directory name, e.g., 'bots')
            description: Short package description
            version: Initial version number
            category: Package category (core, experimental, legacy)
            requires_docs_build: Whether package needs documentation builds

        Returns:
            True if successful, False otherwise
        """
        pypi_name = f"dataknobs-{name}"

        print(f"\n📦 Creating new DataKnobs package: {name}")
        print(f"   PyPI name: {pypi_name}")
        print(f"   Description: {description}")
        print(f"   Version: {version}")
        print(f"   Category: {category}")
        print(f"   Docs build: {requires_docs_build}")
        print()

        # Create package structure
        if not self.create_package_structure(name, description, version):
            return False

        # Update package registry
        self.update_package_registry(
            name, pypi_name, description, version, category, requires_docs_build
        )

        # Update GitHub workflows
        if requires_docs_build:
            self.update_workflow_docs(name)

        if category != "experimental":
            self.update_release_workflow(name, category)

        # Update README
        self.update_readme(name, pypi_name, description, category)

        # Update root pyproject.toml
        self.update_pyproject_toml(name, pypi_name)

        print(f"\n✅ Package '{name}' created successfully!")
        print(f"\n📋 Next steps (see docs/development/new-package-checklist.md):")
        print(f"   1. Implement package code in packages/{name}/src/dataknobs_{name}/")
        print(f"   2. Write tests in packages/{name}/tests/")
        print(f"   3. Add documentation to packages/{name}/docs/")
        if requires_docs_build:
            print(f"   4. Add package to mkdocs.yml")
            print(f"   5. Create docs/packages/{name}/ documentation")
        print(f"   6. Run: uv sync --all-packages")
        print(f"   7. Run: ./bin/validate-package-references.py")
        print(f"   8. Run tests: uv run pytest packages/{name}/tests/ -v")

        return True

    def _generate_pyproject_toml(self, name: str, description: str, version: str) -> str:
        """Generate pyproject.toml content for new package."""
        return f'''[project]
name = "dataknobs-{name}"
version = "{version}"
description = "{description}"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "dataknobs-common>=1.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/dataknobs_{name}"]
'''

    def _generate_init_py(self, name: str, version: str) -> str:
        """Generate __init__.py content for new package."""
        module_name = f"dataknobs_{name}"
        return f'''"""DataKnobs {name.capitalize()} package.

{name.capitalize()} functionality for the DataKnobs ecosystem.
"""

__version__ = "{version}"

__all__ = [
    "__version__",
]
'''

    def _generate_package_readme(self, name: str, description: str) -> str:
        """Generate README.md for new package."""
        return f'''# dataknobs-{name}

{description}

## Installation

```bash
pip install dataknobs-{name}
```

## Usage

```python
from dataknobs_{name} import ...

# Your usage example here
```

## Documentation

See the [full documentation](../../docs/packages/{name}/) for detailed usage guides and API reference.

## Development

```bash
# Install package in development mode
cd packages/{name}
uv pip install -e .

# Run tests
uv run pytest tests/ -v

# Run type checking
uv run mypy src/
```

## License

See the [LICENSE](../../LICENSE) file for details.
'''

    def _generate_test_file(self, name: str) -> str:
        """Generate initial test file."""
        module_name = f"dataknobs_{name}"
        return f'''"""Tests for dataknobs_{name} package."""

import pytest


def test_import():
    """Test that the package can be imported."""
    import {module_name}
    assert {module_name}.__version__


def test_placeholder():
    """Placeholder test - replace with actual tests."""
    assert True
'''


def main():
    parser = argparse.ArgumentParser(
        description="Create a new DataKnobs package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "name",
        help="Package name (e.g., 'ml', 'agents', 'tools')"
    )

    parser.add_argument(
        "-d", "--description",
        required=True,
        help="Short package description"
    )

    parser.add_argument(
        "-v", "--version",
        default="0.1.0",
        help="Initial version number (default: 0.1.0)"
    )

    parser.add_argument(
        "-c", "--category",
        choices=["core", "experimental", "legacy"],
        default="core",
        help="Package category (default: core)"
    )

    parser.add_argument(
        "--no-docs",
        action="store_true",
        help="Package does not require documentation builds"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    # Validate package name
    name = args.name.lower()
    if not name.isidentifier() or name.startswith("_"):
        print(f"❌ Invalid package name: {name}")
        print("   Package name must be a valid Python identifier")
        sys.exit(1)

    repo_root = Path(__file__).parent.parent

    creator = PackageCreator(repo_root, dry_run=args.dry_run)

    success = creator.create_package(
        name=name,
        description=args.description,
        version=args.version,
        category=args.category,
        requires_docs_build=not args.no_docs
    )

    if args.dry_run:
        print("\n[DRY RUN] No changes were made. Run without --dry-run to create the package.")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

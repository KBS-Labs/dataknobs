#!/usr/bin/env python3
"""Validate that all packages are properly referenced across the codebase.

This script checks:
- GitHub workflows reference all packages that require docs build
- Release workflow has all packages in choices
- README.md mentions all non-deprecated packages
- pyproject.toml includes all packages
"""

import json
import sys
from pathlib import Path


def load_registry():
    """Load the package registry."""
    registry_path = Path(__file__).parent.parent / ".dataknobs" / "packages.json"
    with open(registry_path) as f:
        return json.load(f)


def check_workflow_docs(registry, repo_root):
    """Check if docs workflows install all required packages."""
    errors = []

    doc_packages = [p for p in registry["packages"] if p.get("requires_docs_build", False)]

    workflow_files = [
        ".github/workflows/docs.yml",
        ".github/workflows/quality-validation.yml"
    ]

    for workflow_file in workflow_files:
        workflow_path = repo_root / workflow_file
        if not workflow_path.exists():
            errors.append(f"❌ Workflow file not found: {workflow_file}")
            continue

        content = workflow_path.read_text()

        for pkg in doc_packages:
            expected = f"packages/{pkg['name']}"
            if expected not in content:
                errors.append(
                    f"❌ {workflow_file}: Package '{pkg['name']}' not found in install steps"
                )

    return errors


def check_release_workflow(registry, repo_root):
    """Check if release workflow has all packages in choices."""
    errors = []

    releasable_packages = [
        p for p in registry["packages"]
        if not p.get("deprecated", False) and p["category"] != "experimental"
    ]

    workflow_path = repo_root / ".github/workflows/release.yml"
    if not workflow_path.exists():
        errors.append("❌ Release workflow not found")
        return errors

    content = workflow_path.read_text()

    for pkg in releasable_packages:
        # Check if package name appears in options section
        if f"- {pkg['name']}" not in content:
            errors.append(
                f"❌ release.yml: Package '{pkg['name']}' not in workflow choices"
            )

    return errors


def check_readme(registry, repo_root):
    """Check if README mentions all non-deprecated packages."""
    errors = []

    listed_packages = [
        p for p in registry["packages"]
        if not p.get("deprecated", False) and p["category"] == "core"
    ]

    readme_path = repo_root / "README.md"
    if not readme_path.exists():
        errors.append("❌ README.md not found")
        return errors

    content = readme_path.read_text()

    for pkg in listed_packages:
        if pkg["pypi_name"] not in content:
            errors.append(
                f"❌ README.md: Package '{pkg['pypi_name']}' not mentioned"
            )

    return errors


def check_pyproject_toml(registry, repo_root):
    """Check if pyproject.toml includes all packages."""
    errors = []

    pyproject_path = repo_root / "pyproject.toml"
    if not pyproject_path.exists():
        errors.append("❌ pyproject.toml not found")
        return errors

    content = pyproject_path.read_text()

    for pkg in registry["packages"]:
        if pkg["pypi_name"] not in content:
            errors.append(
                f"⚠️  pyproject.toml: Package '{pkg['pypi_name']}' not found"
            )

    return errors


def main():
    repo_root = Path(__file__).parent.parent
    registry = load_registry()

    print("🔍 Validating package references across codebase...\n")

    all_errors = []

    # Run checks
    checks = [
        ("GitHub Workflows (docs)", check_workflow_docs),
        ("Release Workflow", check_release_workflow),
        ("README.md", check_readme),
        ("pyproject.toml", check_pyproject_toml),
    ]

    for check_name, check_func in checks:
        print(f"Checking {check_name}...")
        errors = check_func(registry, repo_root)
        if errors:
            all_errors.extend(errors)
            for error in errors:
                print(f"  {error}")
        else:
            print(f"  ✅ All packages properly referenced")
        print()

    # Summary
    if all_errors:
        print(f"\n❌ Found {len(all_errors)} issue(s)")
        print("\nTo fix these issues, ensure all packages in .dataknobs/packages.json")
        print("are properly referenced in the relevant files.")
        sys.exit(1)
    else:
        print("✅ All package references are consistent!")
        sys.exit(0)


if __name__ == "__main__":
    main()

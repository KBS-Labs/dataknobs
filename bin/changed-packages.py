#!/usr/bin/env python3
"""Detect changed packages and their dependents for targeted testing.

Analyzes git changes to determine which packages need testing,
computing the transitive closure of dependents via the dependency graph.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Package dependency graph: package -> list of packages it depends on
DEPENDENCIES: dict[str, list[str]] = {
    "common": [],
    "structures": ["common"],
    "config": ["common"],
    "utils": ["common", "structures"],
    "xization": ["common", "structures", "utils"],
    "data": ["common", "utils", "config"],
    "fsm": ["common", "data", "structures", "utils", "config"],
    "llm": ["common", "config", "data"],
    "bots": ["common", "config", "llm", "data", "xization"],
}

# All valid package names
ALL_PACKAGES = sorted(DEPENDENCIES.keys())

# Files/dirs that trigger testing all packages
GLOBAL_TRIGGERS = [
    "pyproject.toml",
    "uv.lock",
    "conftest.py",
]

# Prefixes that trigger testing all packages
GLOBAL_PREFIXES = [
    "bin/",
]

# Patterns that indicate docs changes
DOCS_PATTERNS = [
    "docs/",
    "mkdocs.yml",
]


def _run_git(*args: str) -> list[str]:
    """Run a git command and return non-empty output lines."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except FileNotFoundError:
        return []


def get_changed_files(base_ref: str) -> list[str]:
    """Get all changed files: committed on branch, staged, and unstaged."""
    files: set[str] = set()

    # Changes committed on branch vs base
    files.update(_run_git("diff", "--name-only", f"{base_ref}...HEAD"))

    # Staged changes
    files.update(_run_git("diff", "--name-only", "--cached"))

    # Unstaged changes
    files.update(_run_git("diff", "--name-only"))

    # Untracked files (new files not yet staged)
    files.update(_run_git("ls-files", "--others", "--exclude-standard"))

    return sorted(files)


def build_reverse_graph() -> dict[str, list[str]]:
    """Build reverse dependency graph: package -> packages that depend on it."""
    reverse: dict[str, list[str]] = {pkg: [] for pkg in DEPENDENCIES}
    for pkg, deps in DEPENDENCIES.items():
        for dep in deps:
            reverse[dep].append(pkg)
    return reverse


def get_transitive_dependents(packages: set[str]) -> set[str]:
    """Compute transitive closure of all packages that depend on the given set."""
    reverse = build_reverse_graph()
    result = set(packages)
    queue = list(packages)

    while queue:
        pkg = queue.pop()
        for dependent in reverse.get(pkg, []):
            if dependent not in result:
                result.add(dependent)
                queue.append(dependent)

    return result


def map_files_to_packages(files: list[str]) -> tuple[set[str], bool, bool]:
    """Map changed files to affected packages.

    Returns:
        (directly_changed_packages, docs_changed, all_packages_triggered)
    """
    changed_packages: set[str] = set()
    docs_changed = False
    all_triggered = False

    for filepath in files:
        # Check for global triggers
        if filepath in GLOBAL_TRIGGERS:
            all_triggered = True
            continue

        if any(filepath.startswith(prefix) for prefix in GLOBAL_PREFIXES):
            all_triggered = True
            continue

        # Check for docs changes
        if any(filepath.startswith(pattern) for pattern in DOCS_PATTERNS):
            docs_changed = True
            continue

        # Check for package-specific docs
        if "/docs/" in filepath and filepath.startswith("packages/"):
            docs_changed = True

        # Map to package
        if filepath.startswith("packages/"):
            parts = filepath.split("/")
            if len(parts) >= 2:
                pkg_name = parts[1]
                if pkg_name in DEPENDENCIES:
                    changed_packages.add(pkg_name)

    return changed_packages, docs_changed, all_triggered


def detect_changes(base_ref: str = "main") -> dict:
    """Detect changed packages and docs status.

    Returns dict with:
        packages: sorted list of package names that need testing
        docs_changed: whether docs-related files changed
        directly_changed: packages with direct file changes
        mode: "all" if global trigger hit, "changed" otherwise
    """
    changed_files = get_changed_files(base_ref)

    if not changed_files:
        return {
            "packages": [],
            "docs_changed": False,
            "directly_changed": [],
            "mode": "none",
        }

    directly_changed, docs_changed, all_triggered = map_files_to_packages(changed_files)

    if all_triggered:
        return {
            "packages": ALL_PACKAGES,
            "docs_changed": docs_changed or all_triggered,
            "directly_changed": sorted(directly_changed),
            "mode": "all",
        }

    # Compute transitive dependents
    all_affected = get_transitive_dependents(directly_changed)

    # Filter to only packages that actually exist
    packages = sorted(pkg for pkg in all_affected if pkg in DEPENDENCIES)

    return {
        "packages": packages,
        "docs_changed": docs_changed,
        "directly_changed": sorted(directly_changed),
        "mode": "changed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect changed packages for targeted testing"
    )
    parser.add_argument(
        "--base-ref",
        default="main",
        help="Git ref to compare against (default: main)",
    )
    args = parser.parse_args()

    result = detect_changes(args.base_ref)
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

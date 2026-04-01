#!/usr/bin/env python3
"""Detect changed packages and their dependents for targeted testing.

Analyzes git changes to determine which packages need testing,
computing the transitive closure of dependents via the dependency graph.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# Root of the repository
_ROOT = Path(__file__).resolve().parent.parent
_PACKAGES_DIR = _ROOT / "packages"

# Regex to extract dataknobs-<name> from dependency strings like:
#   "dataknobs-common>=1.0.1",
_DK_DEP_RE = re.compile(r'"dataknobs-([a-z]+)')


def discover_dependencies() -> dict[str, list[str]]:
    """Build the dependency graph by parsing each package's pyproject.toml.

    Returns a dict mapping package short name to the list of internal
    dataknobs package short names it depends on.
    """
    deps: dict[str, list[str]] = {}
    for pyproject in sorted(_PACKAGES_DIR.glob("*/pyproject.toml")):
        pkg_name = pyproject.parent.name
        internal_deps: list[str] = []
        in_deps_section = False
        for line in pyproject.read_text().splitlines():
            stripped = line.strip()
            if stripped == "dependencies = [":
                in_deps_section = True
                continue
            if in_deps_section:
                if stripped == "]":
                    break
                m = _DK_DEP_RE.search(stripped)
                if m:
                    dep_name = m.group(1)
                    if dep_name != pkg_name:  # skip self-references
                        internal_deps.append(dep_name)
        deps[pkg_name] = sorted(internal_deps)
    return deps


# Discover at import time — this script is short-lived (CLI tool)
DEPENDENCIES = discover_dependencies()

# All valid package names
ALL_PACKAGES = sorted(DEPENDENCIES.keys())

# Files/dirs that trigger testing all packages
GLOBAL_TRIGGERS = [
    "pyproject.toml",
    "uv.lock",
    "conftest.py",
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


def _resolve_base_ref(base_ref: str) -> str:
    """Resolve the base ref, preferring the remote-tracking branch.

    When the user passes "main", we want "origin/main" so that change
    detection works even when the local branch is behind the remote.
    Falls back to the original ref if the remote variant doesn't exist.
    """
    # If already a remote ref or explicit path, use as-is
    if "/" in base_ref:
        return base_ref

    # Try origin/<ref> first
    remote_ref = f"origin/{base_ref}"
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", remote_ref],
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return remote_ref

    return base_ref


def get_changed_files(base_ref: str) -> list[str]:
    """Get all changed files: committed on branch, staged, and unstaged."""
    files: set[str] = set()

    resolved_ref = _resolve_base_ref(base_ref)

    # Changes committed on branch vs base
    files.update(_run_git("diff", "--name-only", f"{resolved_ref}...HEAD"))

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


def detect_changes(base_ref: str = "main") -> dict[str, Any]:
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
            "docs_changed": docs_changed,
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

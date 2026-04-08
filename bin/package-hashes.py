#!/usr/bin/env python3
"""Compute per-package content hashes for quality artifact validation.

Produces deterministic SHA-256 hashes of each package's quality-relevant
files (source, tests, pyproject.toml). Used by run-quality-checks.sh to
stamp artifacts and by validate-quality-artifacts.sh to detect staleness.

Reuses the dependency graph from changed-packages.py to compute the
transitive "dirty set" of packages needing re-validation.
"""

import argparse
import hashlib
import importlib.util
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_PACKAGES_DIR = _ROOT / "packages"
_ARTIFACTS_DIR = _ROOT / ".quality-artifacts"

# Quality-relevant file patterns per package (relative to package dir)
_HASH_PATTERNS = [
    ("src", "**/*.py"),
    ("tests", "**/*.py"),
    (".", "pyproject.toml"),
]

# Lines matching these patterns are stripped before hashing because they
# change during releases but do not affect code quality.
_VERSION_LINE_RE = re.compile(
    r'^(?:version\s*=\s*"[^"]*"|__version__\s*=\s*"[^"]*")\s*$'
)

# Increment when the hashing algorithm changes (e.g., adding version stripping).
# A mismatch between stored and current version means hashes are incomparable
# and the artifacts should be treated as needing a fresh baseline.
_HASH_ALGORITHM_VERSION = 2


def _load_changed_packages() -> Any:
    """Import changed-packages.py (hyphenated name requires importlib)."""
    script_path = Path(__file__).resolve().parent / "changed-packages.py"
    spec = importlib.util.spec_from_file_location("changed_packages", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load {script_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load dependency graph utilities
_changed_packages = _load_changed_packages()
ALL_PACKAGES: list[str] = _changed_packages.ALL_PACKAGES
get_transitive_dependents = _changed_packages.get_transitive_dependents


def compute_package_hash(package_name: str) -> str:
    """Compute a deterministic SHA-256 hash of a package's quality-relevant files.

    Hashes relative file paths and file contents to detect both content
    changes and file renames/additions/deletions.
    """
    pkg_dir = _PACKAGES_DIR / package_name
    if not pkg_dir.is_dir():
        return hashlib.sha256(b"missing").hexdigest()

    hasher = hashlib.sha256()
    all_files: list[Path] = []

    for subdir, glob_pattern in _HASH_PATTERNS:
        target = pkg_dir / subdir if subdir != "." else pkg_dir
        if target.exists():
            all_files.extend(target.glob(glob_pattern))

    # Sort by relative path for cross-platform determinism
    all_files.sort(key=lambda f: str(f.relative_to(pkg_dir)))

    for filepath in all_files:
        rel_path = str(filepath.relative_to(pkg_dir))
        content = filepath.read_bytes()

        # Strip version-only lines so release bumps don't dirty packages.
        # Version strings (in pyproject.toml and __init__.py) change during
        # releases but have no effect on code quality or test outcomes.
        filtered_lines = [
            line
            for line in content.decode("utf-8", errors="surrogateescape").splitlines(keepends=True)
            if not _VERSION_LINE_RE.match(line.strip())
        ]
        filtered_content = "".join(filtered_lines).encode("utf-8")

        # Hash path + content with null-byte separators to avoid ambiguity
        hasher.update(rel_path.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(filtered_content)
        hasher.update(b"\x00")

    return hasher.hexdigest()


def compute_all_hashes() -> dict[str, str]:
    """Compute content hashes for all packages."""
    return {pkg: compute_package_hash(pkg) for pkg in ALL_PACKAGES}


def validate_artifacts() -> dict[str, Any]:
    """Compare current content hashes against stored artifact hashes.

    Uses the dependency graph to compute the transitive dirty set:
    any package whose content changed, plus all packages that depend
    on a changed package.

    Returns a structured result dict with validity status and details.
    """
    summary_path = _ARTIFACTS_DIR / "quality-summary.json"

    if not summary_path.exists():
        return {
            "valid": False,
            "error": "quality-summary.json not found",
            "changed_packages": [],
            "dirty_packages": [],
        }

    summary = json.loads(summary_path.read_text())
    stored_hashes = summary.get("package_hashes", {})

    if not stored_hashes:
        return {
            "valid": True,
            "warning": "No package_hashes in quality-summary.json — skipping hash validation",
            "changed_packages": [],
            "dirty_packages": [],
        }

    stored_algorithm = stored_hashes.pop("_algorithm_version", 1)
    if stored_algorithm != _HASH_ALGORITHM_VERSION:
        return {
            "valid": True,
            "warning": (
                f"Hash algorithm changed (stored: v{stored_algorithm}, "
                f"current: v{_HASH_ALGORITHM_VERSION}) — skipping hash validation"
            ),
            "changed_packages": [],
            "dirty_packages": [],
        }

    current_hashes = compute_all_hashes()

    # Find packages whose content has changed
    changed: set[str] = set()
    for pkg in current_hashes:
        if current_hashes[pkg] != stored_hashes.get(pkg):
            changed.add(pkg)

    # Compute transitive dirty set (changed + all downstream dependents)
    dirty = get_transitive_dependents(changed) if changed else set()

    overall_status = summary.get("overall_status", "")
    status_ok = overall_status in ("PASS", "PASS_WITH_SKIPS")

    return {
        "valid": len(dirty) == 0 and status_ok,
        "changed_packages": sorted(changed),
        "dirty_packages": sorted(dirty),
        "status_ok": status_ok,
        "overall_status": overall_status,
    }


def cmd_compute() -> None:
    """Print per-package content hashes as JSON (with algorithm version)."""
    result = compute_all_hashes()
    result["_algorithm_version"] = _HASH_ALGORITHM_VERSION
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


def cmd_validate(*, use_json: bool = False) -> None:
    """Validate that artifacts match current source content."""
    result = validate_artifacts()

    if use_json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        if result.get("error"):
            logger.error("Validation error: %s", result["error"])
        elif result.get("warning"):
            logger.warning("%s", result["warning"])
        elif result["valid"]:
            if result["dirty_packages"]:
                logger.info("All dirty packages have been tested")
            else:
                logger.info("All packages unchanged since last quality run")
        else:
            logger.error("Quality artifacts are stale")
            if result["changed_packages"]:
                logger.error("Changed packages: %s", ", ".join(result["changed_packages"]))
            if result["dirty_packages"]:
                logger.error(
                    "Packages needing re-validation: %s",
                    ", ".join(result["dirty_packages"]),
                )

    sys.exit(0 if result["valid"] else 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and validate per-package content hashes"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("compute", help="Print per-package content hashes as JSON")

    validate_parser = subparsers.add_parser(
        "validate", help="Validate artifacts against current source content"
    )
    validate_parser.add_argument(
        "--json", action="store_true", dest="use_json",
        help="Output structured JSON result",
    )

    args = parser.parse_args()

    if args.command == "compute":
        cmd_compute()
    elif args.command == "validate":
        cmd_validate(use_json=args.use_json)


if __name__ == "__main__":
    main()

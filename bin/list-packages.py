#!/usr/bin/env python3
"""List packages from the registry in various formats.

Usage:
    python bin/list-packages.py --format yaml          # For GitHub Actions
    python bin/list-packages.py --format choices       # For workflow inputs
    python bin/list-packages.py --format pip           # For pip install commands
    python bin/list-packages.py --format docs-install  # For docs build
    python bin/list-packages.py --category core        # Filter by category
    python bin/list-packages.py --exclude-deprecated   # Exclude deprecated packages
"""

import argparse
import json
from pathlib import Path
from typing import Any

#: One entry from the registry. Left open rather than modelled as a TypedDict:
#: this script only reads a handful of keys, and the registry's shape is owned
#: by .dataknobs/packages.json rather than here — a second declaration of it
#: would be one more thing to keep in step with the file it describes.
Package = dict[str, Any]


def load_registry() -> dict[str, Any]:
    """Load the package registry."""
    registry_path = Path(__file__).parent.parent / ".dataknobs" / "packages.json"
    with open(registry_path) as f:
        result: dict[str, Any] = json.load(f)
        return result


def filter_packages(
    packages: list[Package],
    category: str | None = None,
    exclude_deprecated: bool = False,
    requires_docs: bool | None = None,
) -> list[Package]:
    """Filter packages based on criteria."""
    filtered = packages

    if category:
        filtered = [p for p in filtered if p["category"] == category]

    if exclude_deprecated:
        filtered = [p for p in filtered if not p.get("deprecated", False)]

    if requires_docs is not None:
        filtered = [p for p in filtered if p.get("requires_docs_build", False) == requires_docs]

    return filtered


def format_yaml_list(packages: list[Package]) -> str:
    """Format as YAML list for GitHub Actions matrix."""
    names = [p["name"] for p in packages]
    # GitHub Actions matrix format
    return json.dumps(names)


def format_choices(packages: list[Package]) -> None:
    """Format as workflow input choices."""
    names = [p["name"] for p in packages]
    for name in sorted(names):
        print(f"          - {name}")


def format_pip_install(packages: list[Package]) -> str:
    """Format as pip install command."""
    names = [p["pypi_name"] for p in packages]
    return " ".join(names)


def format_uv_install(packages: list[Package]) -> None:
    """Format as uv pip install commands for docs build."""
    for pkg in packages:
        print(f"        uv pip install --system -e packages/{pkg['name']}")


def format_table_row(packages: list[Package]) -> None:
    """Format as markdown table rows."""
    for pkg in packages:
        pypi = pkg["pypi_name"]
        desc = pkg["description"]
        version = pkg["version"]
        name = pkg["name"]
        print(f"| [{pypi}](packages/{name}/index.md) | {desc} | {version} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="List packages from registry")
    parser.add_argument(
        "--format",
        choices=["yaml", "choices", "pip", "uv-install", "table"],
        default="yaml",
        help="Output format"
    )
    parser.add_argument(
        "--category",
        help="Filter by category (core, experimental, legacy)"
    )
    parser.add_argument(
        "--exclude-deprecated",
        action="store_true",
        help="Exclude deprecated packages"
    )
    parser.add_argument(
        "--requires-docs",
        action="store_true",
        help="Only packages that require docs build"
    )

    args = parser.parse_args()

    registry = load_registry()
    packages = filter_packages(
        registry["packages"],
        category=args.category,
        exclude_deprecated=args.exclude_deprecated,
        requires_docs=args.requires_docs if args.requires_docs else None
    )

    if args.format == "yaml":
        print(format_yaml_list(packages))
    elif args.format == "choices":
        format_choices(packages)
    elif args.format == "pip":
        print(format_pip_install(packages))
    elif args.format == "uv-install":
        format_uv_install(packages)
    elif args.format == "table":
        format_table_row(packages)


if __name__ == "__main__":
    main()

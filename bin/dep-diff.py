#!/usr/bin/env python3
"""Compare uv.lock dependency versions against a git ref.

Shows updated, added, and removed packages between the current
working tree's uv.lock and the version at a given git ref (default: main).

Usage:
    bin/dep-diff.py [ref]        Compare against ref (default: main)
    bin/dep-diff.py --staged      Compare staged uv.lock against HEAD
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class VersionChange:
    old: str
    new: str

    @property
    def is_major(self) -> bool:
        """True if the major version component changed."""
        old_major = self.old.split(".")[0]
        new_major = self.new.split(".")[0]
        return old_major != new_major


def parse_packages(text: str) -> dict[str, str]:
    """Parse [[package]] blocks from uv.lock content into {name: version}."""
    packages: dict[str, str] = {}
    name = None
    for line in text.splitlines():
        m = re.match(r'^name\s*=\s*"(.+)"', line)
        if m:
            name = m.group(1)
            continue
        m = re.match(r'^version\s*=\s*"(.+)"', line)
        if m and name is not None:
            packages[name] = m.group(1)
            name = None
    return packages


def git_show(ref: str, path: str) -> str | None:
    """Return file contents at a given git ref, or None if it doesn't exist."""
    try:
        result = subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def read_file(path: str) -> str | None:
    """Read a file from the working tree."""
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return None


def git_staged(path: str) -> str | None:
    """Return the staged (index) version of a file."""
    try:
        result = subprocess.run(
            ["git", "show", f":{path}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None


# --- Terminal colors ---

def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


if _supports_color():
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    CYAN = "\033[0;36m"
    NC = "\033[0m"
else:
    BOLD = DIM = RED = GREEN = YELLOW = CYAN = NC = ""


def main() -> None:
    staged = False
    ref = "main"

    args = sys.argv[1:]
    if "--staged" in args:
        staged = True
        args.remove("--staged")
    if args:
        ref = args[0]

    lock_path = "uv.lock"

    # Get base version
    if staged:
        base_text = git_show("HEAD", lock_path)
        current_text = git_staged(lock_path)
        label = "staged vs HEAD"
    else:
        base_text = git_show(ref, lock_path)
        current_text = read_file(lock_path)
        label = f"current vs {ref}"

    if base_text is None:
        print(f"{RED}Error:{NC} uv.lock not found at {'HEAD' if staged else ref}")
        sys.exit(1)
    if current_text is None:
        print(f"{RED}Error:{NC} uv.lock not found in {'index' if staged else 'working tree'}")
        sys.exit(1)

    base_pkgs = parse_packages(base_text)
    current_pkgs = parse_packages(current_text)

    all_names = sorted(set(base_pkgs) | set(current_pkgs))

    updated: list[tuple[str, VersionChange]] = []
    added: list[tuple[str, str]] = []
    removed: list[tuple[str, str]] = []

    for name in all_names:
        in_base = name in base_pkgs
        in_current = name in current_pkgs
        if in_base and in_current:
            if base_pkgs[name] != current_pkgs[name]:
                updated.append((name, VersionChange(base_pkgs[name], current_pkgs[name])))
        elif in_current:
            added.append((name, current_pkgs[name]))
        else:
            removed.append((name, base_pkgs[name]))

    if not updated and not added and not removed:
        print(f"{GREEN}No dependency changes{NC} ({label})")
        return

    print(f"{BOLD}Dependency changes{NC} ({label}):\n")

    if updated:
        print(f"{BOLD}{CYAN}Updated ({len(updated)}):{NC}")
        name_width = max(len(n) for n, _ in updated)
        old_width = max(len(c.old) for _, c in updated)
        for name, change in updated:
            marker = f"  {RED}MAJOR{NC}" if change.is_major else ""
            print(
                f"  {name:<{name_width}}  "
                f"{DIM}{change.old:>{old_width}}{NC}"
                f" -> "
                f"{GREEN}{change.new}{NC}"
                f"{marker}"
            )
        print()

    if added:
        print(f"{BOLD}{GREEN}Added ({len(added)}):{NC}")
        name_width = max(len(n) for n, _ in added)
        for name, version in added:
            print(f"  {name:<{name_width}}  {GREEN}{version}{NC}")
        print()

    if removed:
        print(f"{BOLD}{RED}Removed ({len(removed)}):{NC}")
        name_width = max(len(n) for n, _ in removed)
        for name, version in removed:
            print(f"  {name:<{name_width}}  {RED}{version}{NC}")
        print()

    # Summary line
    parts = []
    if updated:
        parts.append(f"{len(updated)} updated")
    if added:
        parts.append(f"{len(added)} added")
    if removed:
        parts.append(f"{len(removed)} removed")
    major_count = sum(1 for _, c in updated if c.is_major)
    if major_count:
        parts.append(f"{RED}{major_count} major version bump{'s' if major_count > 1 else ''}{NC}")
    print(f"{BOLD}Total:{NC} {', '.join(parts)}")


if __name__ == "__main__":
    main()

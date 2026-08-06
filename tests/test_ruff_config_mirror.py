"""Reproduce-first guard for the ruff ignore list's six copies.

The root ``pyproject.toml`` is authoritative — every script lints with
``--config <root>``. The five per-package ``[tool.ruff]`` sections exist for
IDE and hierarchical invocations, and they disagree with root on thousands of
findings by design: reconciling them is a separate lint-policy question.

What must *not* differ is the set of deliberately declined modernizations. A
package missing one would report a finding the gate does not have, and the
obvious way to quiet it — deleting the rule from root — silently reverses a
decision recorded with a reason. Only the rule identifiers are compared; the
prose beside them is abbreviated per package on purpose.

This is not the general reconciliation, and it should not grow into it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Declined because each changes a public API rather than syntax. Named here so
#: adding a decline to root without mirroring it fails, rather than drifting.
DECLINED = frozenset({"ASYNC109", "UP040", "UP042", "UP046", "UP047"})


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _ruff_configs() -> dict[Path, dict]:
    """Every ``pyproject.toml`` that declares its own ``[tool.ruff]``."""
    found = {}
    for path in [ROOT / "pyproject.toml", *sorted(ROOT.glob("packages/*/pyproject.toml"))]:
        ruff = tomllib.loads(path.read_text(encoding="utf-8")).get("tool", {}).get("ruff")
        if ruff is not None:
            found[path] = ruff
    return found


def _ignores(ruff: dict) -> set[str]:
    return set(ruff.get("lint", {}).get("ignore", []) or ruff.get("ignore", []) or [])


def test_root_declines_exactly_the_recorded_set():
    """Root is authoritative, so drift there is the one that changes the gate."""
    root = _ignores(_ruff_configs()[ROOT / "pyproject.toml"])
    missing = sorted(DECLINED - root)
    assert not missing, (
        "Root no longer declines "
        f"{missing} — if that is intended, update DECLINED and sweep the findings"
    )


def test_every_package_mirrors_the_declines():
    """A package missing one reports findings the authoritative config does not."""
    violations = [
        f"{_rel(path)}: missing {sorted(DECLINED - _ignores(ruff))}"
        for path, ruff in _ruff_configs().items()
        if path != ROOT / "pyproject.toml" and DECLINED - _ignores(ruff)
    ]

    assert not violations, "Declined rules are not mirrored:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


def test_every_declined_rule_carries_a_reason_in_root():
    """A bare identifier is a decision with the reason lost.

    Checked only in root: the per-package copies deliberately abbreviate, and
    root is where the full rationale is kept.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    violations = []
    for rule in sorted(DECLINED):
        line = next((ln for ln in text.splitlines() if f'"{rule}",' in ln), None)
        if line is None:
            violations.append(f"{rule}: not found in root pyproject.toml")
        elif "#" not in line:
            violations.append(f"{rule}: declined with no reason on its line")

    assert not violations, "Declines missing a rationale:\n" + "\n".join(
        f"  - {v}" for v in violations
    )

"""Reproduce-first guard for the ruff configuration's six copies.

The root ``pyproject.toml`` is authoritative — every script lints with
``--config <root>``. The five per-package ``[tool.ruff]`` sections exist for
IDE and hierarchical invocations, and they disagree with root by thousands of
findings. Which configuration applies is therefore decided by *the command*,
not the repo, and the divergence is invisible from either side alone.

Three things are checked, in descending order of how settled they are:

1. The set of deliberately declined modernizations is mirrored everywhere. A
   package missing one reports a finding the gate does not have, and the
   obvious way to quiet it — deleting the rule from root — silently reverses a
   decision recorded with a reason.
2. No package selects a rule family root does not. This holds today, so it is
   a hard error: it is the direction that would newly diverge.
3. Everything else a package enables that the gate does not is *reported*, not
   asserted. That gap is real and known; closing it is a lint-policy decision
   about whether five packages should be stricter than the gate, which is not
   this guard's call to force.

Reconciling the divergence is out of scope here. Naming it is the point.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tomllib
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Declined because each changes a public API rather than syntax. Named here so
#: adding a decline to root without mirroring it fails, rather than drifting.
DECLINED = frozenset({"ASYNC109", "UP040", "UP042", "UP046", "UP047"})

#: ``ruff check --show-settings`` prints the fully resolved rule set for the
#: config that applies to a path, one ``name (CODE),`` per line.
ENABLED_BLOCK_RE = re.compile(r"^linter\.rules\.enabled = \[(.*?)^\]", re.DOTALL | re.MULTILINE)
RULE_CODE_RE = re.compile(r"\(([A-Z]+\d+)\)")


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


def _selects(ruff: dict) -> set[str]:
    return set(ruff.get("lint", {}).get("select", []) or ruff.get("select", []) or [])


def _package_configs() -> dict[Path, dict]:
    return {p: r for p, r in _ruff_configs().items() if p != ROOT / "pyproject.toml"}


def _enabled_rules(probe: Path, config: Path | None) -> frozenset[str]:
    """The rule codes ruff resolves as enabled for ``probe``.

    Asking ruff beats reimplementing its selector semantics, and not by a
    small margin — a hand-rolled prefix comparison gets this wrong three
    separate ways on this repo's own configs. ``ERA001`` looks like it falls
    under a selected ``E``; it does not, because ``E`` names pycodestyle and
    ``ERA`` names eradicate. ``TC001`` looks like it falls under nothing,
    because the family is still spelled ``TCH`` in these configs. And
    ``D400`` / ``D401`` / ``D404`` are off in every config regardless of what
    ``select`` and ``ignore`` say, because ``convention = "google"`` disables
    them. Each of those errors is silent, and two of them point the wrong way.
    """
    proc = subprocess.run(
        [
            "ruff",
            "check",
            "--show-settings",
            *(("--config", str(config)) if config else ()),
            str(probe),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    block = ENABLED_BLOCK_RE.search(proc.stdout)
    assert block is not None, (
        f"ruff --show-settings did not report a rule set for {_rel(probe)}"
        f"{f' under {_rel(config)}' if config else ''}:\n{proc.stdout}\n{proc.stderr}"
    )
    return frozenset(RULE_CODE_RE.findall(block.group(1)))


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


def test_no_package_selects_a_family_root_does_not():
    """The direction that is aligned today, held as a hard error.

    Every package's ``select`` is a subset of root's, so a package adding a
    family the gate does not run is a *new* divergence rather than an existing
    one — and it arrives the same way all of these did, as one plausible line
    in one file. Compared on the declared selectors rather than the resolved
    rule set, because that is the edit a reviewer sees.
    """
    root_select = _selects(_ruff_configs()[ROOT / "pyproject.toml"])
    violations = [
        f"{_rel(path)}: selects {sorted(extra)}, which root does not"
        for path, ruff in _package_configs().items()
        if (extra := _selects(ruff) - root_select)
    ]

    assert not violations, (
        "Package lint policy has diverged from the authoritative config:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_report_rules_packages_enable_that_the_gate_does_not():
    """Reports the remaining divergence; deliberately does not fail on it.

    This is the one thing this guard checks that is not green today, and the
    reason it warns rather than asserts is that the fix is not a fix. Nearly
    every gap is a rule root ignores with a recorded reason — ``UP006``,
    ``UP007``, ``UP035`` and the rest of the modernization families that were
    declined because converting them changes a public API. "Closing" the gap
    by adopting the package's stricter view would reverse those decisions
    silently, and by adding the ignores to five more files would leave five
    files to keep in step forever. Whether these sections should exist at all
    is the open question; a red test cannot answer it.

    So the gap is named instead. A warning survives into the gate's captured
    output, which makes the number visible and its growth noticeable, without
    handing anyone a failing check they can only clear by guessing.
    """
    if shutil.which("ruff") is None:
        pytest.skip("ruff is not installed")

    root_config = ROOT / "pyproject.toml"
    gaps: dict[str, list[str]] = {}

    for path in _package_configs():
        probes = sorted((path.parent / "src").rglob("*.py"))
        if not probes:
            continue
        extra = _enabled_rules(probes[0], None) - _enabled_rules(probes[0], root_config)
        if extra:
            gaps[path.parent.name] = sorted(extra)

    if not gaps:
        return

    total = sum(len(rules) for rules in gaps.values())
    detail = "\n".join(f"  - {pkg}: {', '.join(rules)}" for pkg, rules in sorted(gaps.items()))
    warnings.warn(
        f"{total} rules across {len(gaps)} packages are enabled by a package's own "
        f"[tool.ruff] section but not by the authoritative root config, so an IDE "
        f"reports findings the quality gate does not:\n{detail}",
        stacklevel=1,
    )

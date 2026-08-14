"""Reproduce-first guard: the ruff configuration has exactly one copy.

The root ``pyproject.toml`` is authoritative — every script lints with
``--config <root>``. Five packages used to carry their own ``[tool.ruff]``
sections for IDE and hierarchical invocations, and those sections disagreed
with root in both directions at once: they enforced ``E501``, ``I001``,
``SIM117`` and the ``UP0xx`` modernization families the gate had deliberately
declined, while missing ``D209``, ``PLW0108`` and ``PTH118`` that the gate
enforces. Which configuration applied was decided by *the command*, not the
repo, so an editor showed a developer findings no check would ever fail on and
hid findings that would. (``ANN002`` and ``ARG005`` were on that second list
too, until root declined both — a rule family may not sit half-enforced, and
the four hand-written ``# noqa: ARG005`` in ``src`` were the evidence.)

The sections are gone. What replaces them is not a mirror to keep in step —
that was the previous shape, and a mirror is only ever as good as the test that
compares it. It is the absence of a second copy, which is checkable directly.

Three properties, in descending order of how much they would cost to lose:

1. **Hierarchical resolution agrees with the gate**, per package, asked of ruff
   rather than derived. This is the actual goal; the rest are proxies for it.
2. **No package declares ``[tool.ruff]``.** A set-emptiness check — cheap, and
   it cannot silently narrow the way a mirror comparison can, *provided* it is
   asserted to have looked at something, which is the failure mode the previous
   version of this file guarded against in every one of its checks.
3. **Every decline in root carries its reason on its own line.** All of them,
   not the curated five: a bare identifier is a decision with the reason lost,
   and the reason is the only thing that makes it re-litigable. Counted here in
   an earlier draft, which is a figure that goes stale the next time a rule is
   declined and is checked by nothing — the assertion below compares against
   the set it reads rather than against a number written down.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from tests._workspace import ROOT
from tests._workspace import rel as _rel

#: Declined because each changes a public API rather than syntax. Pinned as a
#: set because dropping one from root is the edit that changes the gate, and it
#: looks exactly like tidying an ignore list.
DECLINED = frozenset({"ASYNC109", "UP040", "UP042", "UP046", "UP047"})

#: ``ruff check --show-settings`` prints the fully resolved rule set for the
#: config that applies to a path, one ``name (CODE),`` per line.
ENABLED_BLOCK_RE = re.compile(r"^linter\.rules\.enabled = \[(.*?)^\]", re.DOTALL | re.MULTILINE)
RULE_CODE_RE = re.compile(r"\(([A-Z]+\d+)\)")

#: How this repo resolves ruff. A bare ``ruff`` is not on PATH — it works under
#: the gate only because ``uv run pytest`` prepends ``.venv/bin``, so a bare
#: ``pytest`` would degrade every check below to a silent skip. Every other tool
#: call in the repo is spelled this way for the same reason.
RUFF = ("uv", "run", "ruff")

ROOT_CONFIG = ROOT / "pyproject.toml"


def _package_pyprojects() -> list[Path]:
    """Every package manifest, whether or not it configures ruff."""
    return sorted(ROOT.glob("packages/*/pyproject.toml"))


def _ruff_section(path: Path) -> dict | None:
    return tomllib.loads(path.read_text(encoding="utf-8")).get("tool", {}).get("ruff")


def _declared(ruff: dict, key: str) -> set[str]:
    """A declared selector list, under every spelling ruff accepts.

    ``extend-<key>`` counts. It is the idiomatic way to add to an inherited
    config rather than replace it, so a check that reads only the plain key is
    blind to the single most likely form of the edit it exists to catch.
    """
    lint = ruff.get("lint", {})
    return {
        entry
        for section in (lint, ruff)
        for name in (key, f"extend-{key}")
        for entry in section.get(name, []) or []
    }


def _probe_file(package: Path) -> Path | None:
    """A source file the given package's config would govern."""
    return next(iter(sorted((package.parent / "src").rglob("*.py"))), None)


def _uv_missing() -> bool:
    return shutil.which("uv") is None


def _enabled_rules(probe: Path, config: Path | None) -> frozenset[str]:
    """The rule codes ruff resolves as enabled for ``probe``.

    Asking ruff beats reimplementing its selector semantics, and not by a
    small margin — a hand-rolled prefix comparison gets this wrong three
    separate ways on this repo's own configs. ``ERA001`` looks like it falls
    under a selected ``E``; it does not, because ``E`` names pycodestyle and
    ``ERA`` names eradicate. ``TC001`` looks like it falls under nothing,
    because the family was still spelled ``TCH`` in these configs. And
    ``D400`` / ``D401`` / ``D404`` are off regardless of what ``select`` and
    ``ignore`` say, because ``convention = "google"`` disables them. Each of
    those errors is silent, and two of them point the wrong way.

    Passing ``config=None`` is the whole point of this helper: it asks what an
    editor gets, which is hierarchical discovery from the file upward.
    """
    proc = subprocess.run(
        [
            *RUFF,
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
        f"ruff --show-settings reported no rule set for {_rel(probe)} "
        f"(config={config}):\n{proc.stdout}\n{proc.stderr}"
    )
    return frozenset(RULE_CODE_RE.findall(block.group(1)))


def test_no_package_declares_its_own_ruff_config():
    """The second copy is what this phase removed; this is what keeps it removed.

    Emptiness is the right shape here — a mirror check can only ever be as
    complete as the list of keys it thought to compare. The previous version of
    this file compared ``ignore`` and ``select`` and left ``per-file-ignores``,
    ``line-length`` and ``known-first-party`` unwatched; the last of those had
    in fact drifted, with ``dataknobs_fsm`` first-party to fsm's own section and
    absent from root's. Nothing was wrong with that check except its list.

    The ``manifests`` assertion is not defensive clutter. An emptiness check is
    the one shape that passes hardest when it is broken: rename the packages
    directory, restructure the workspace, typo the glob, and this reports a
    clean single-source config for a tree it never looked at.
    """
    manifests = _package_pyprojects()
    assert manifests, (
        "no packages/*/pyproject.toml found at all — this guard would report "
        "a single-source ruff config for a workspace it did not read"
    )

    declaring = sorted(_rel(path) for path in manifests if _ruff_section(path) is not None)
    assert not declaring, (
        "ruff is configured in more than one place again:\n"
        + "\n".join(f"  - {path}" for path in declaring)
        + "\n\nThe gate lints with --config <root>, so a package section changes "
        "only what editors and bare `ruff` invocations see — which is precisely "
        "how the two came to disagree in both directions. Put the rule in root, "
        "or scope it with a per-file-ignores entry there."
    )


@pytest.mark.parametrize("manifest", _package_pyprojects(), ids=lambda p: p.parent.name)
def test_an_editor_resolves_the_same_rules_the_gate_enforces(manifest: Path):
    """The property the deletion was *for*, asked per package rather than assumed.

    Emptiness above is a proxy: it says there is no second config file. This
    says the thing that second file was breaking — that opening a package source
    in an editor, or running a bare ``ruff check`` on it, resolves the rule set
    the gate will judge it by. Those are different claims. Hierarchical discovery
    could stop agreeing without any package declaring anything, if root's own
    ``[tool.ruff]`` moved or the workspace layout changed.

    Parametrized per package so a failure names the one that diverged instead of
    reporting a set difference over ten trees at once.
    """
    if _uv_missing():
        pytest.skip("uv is not available to resolve ruff")

    probe = _probe_file(manifest)
    if probe is None:
        pytest.skip(f"{manifest.parent.name} has no source under src/ to resolve against")

    discovered = _enabled_rules(probe, None)
    authoritative = _enabled_rules(probe, ROOT_CONFIG)
    assert discovered, f"ruff resolved no rules at all for {_rel(probe)}"

    only_editor = sorted(discovered - authoritative)
    only_gate = sorted(authoritative - discovered)
    assert not (only_editor or only_gate), (
        f"{manifest.parent.name}: hierarchical resolution has diverged from the gate.\n"
        f"  reported by an editor but never failed by the gate: {only_editor}\n"
        f"  enforced by the gate but invisible in an editor:    {only_gate}\n"
        "The first wastes a developer's time on findings no check requires; the "
        "second is worse, because it surfaces at pull-request time on work that "
        "looked clean while it was being written."
    )


def test_root_declines_exactly_the_recorded_set():
    """Root is authoritative, so drift there is the one that changes the gate."""
    ruff = _ruff_section(ROOT_CONFIG)
    assert ruff is not None, "root pyproject.toml declares no [tool.ruff]"

    missing = sorted(DECLINED - _declared(ruff, "ignore"))
    assert not missing, (
        "Root no longer declines "
        f"{missing} — if that is intended, update DECLINED and sweep the findings"
    )


def test_every_declined_rule_carries_a_reason_in_root():
    """Every one of them, not the curated 5 — the reason makes a decline re-litigable.

    Previously this ran over ``DECLINED`` alone, which is five entries chosen
    because they were the ones being argued about. All the rest were
    unwatched, and they are the ones a future reader is most likely to meet
    without context: a bare ``"SIM108",`` says a rule is off and nothing about
    whether that was a judgement or an accident, so the safe move is to leave it
    off forever. Every entry carries a reason today, so this holds the line
    rather than opening a backlog.
    """
    ruff = _ruff_section(ROOT_CONFIG)
    assert ruff is not None, "root pyproject.toml declares no [tool.ruff]"

    declined = _declared(ruff, "ignore")
    assert len(declined) > len(DECLINED), (
        f"root declines only {len(declined)} rules, which is at most the curated "
        "set — this check would then be the narrow one it replaced"
    )

    lines = ROOT_CONFIG.read_text(encoding="utf-8").splitlines()
    violations = []
    for rule in sorted(declined):
        line = next((ln for ln in lines if f'"{rule}",' in ln), None)
        if line is None:
            violations.append(f'{rule}: declared in ignore but not found as a `"{rule}",` line')
        elif "#" not in line:
            violations.append(f"{rule}: declined with no reason on its line")

    assert not violations, (
        f"{len(violations)} of {len(declined)} declines are missing a rationale:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )
